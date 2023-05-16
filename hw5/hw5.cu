#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define BLOCK 128
#define THREAD 1024
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
__constant__ const int n_steps = 200000;
__constant__ const int dt = 60;
__constant__ const double eps = 1e-3;
__constant__ const double G = 6.674e-11;
__device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
__constant__ const double planet_radius = 1e7;
__constant__ const double missile_speed = 1e6;
__device__ double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
// namespace param
#define MIN(a, b) ((a) < (b) ? (a) : (b))

void read_input(const char* filename, int& n, int& planet, int& asteroid,
                std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
                std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++)
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
}

void write_output(const char* filename, double min_dist, int hit_time_step,
                  int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__global__ void run_step(bool isDeviceG, int step, int n,
                         double* qx, double* qy, double* qz,
                         double* ax, double* ay, double* az,
                         const double* m, const char* type) {
    // compute accelerations
    __shared__ double m_shared[THREAD];
    double x, y, z;
    if (threadIdx.x < n) {
        if (type[threadIdx.x] == 5)
            m_shared[threadIdx.x] = (!isDeviceG) ? 0 : m[threadIdx.x];
        else
            m_shared[threadIdx.x] = m[threadIdx.x];
        x = 0, y = 0, z = 0;
    }
    if (threadIdx.x < n && blockIdx.x == 0) {
        ax[threadIdx.x] = 0;
        ay[threadIdx.x] = 0;
        az[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            if (j == i) continue;
            double mi = (type[i] == 5) ? gravity_device_mass(m_shared[i], step * dt) : m_shared[i];
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);
            /* atomicAdd(&ax[i], G * m_shared[j] * dx / dist3);
            atomicAdd(&ay[i], G * m_shared[j] * dy / dist3);
            atomicAdd(&az[i], G * m_shared[j] * dz / dist3); */
            x -= G * mi * dx / dist3;
            y -= G * mi * dy / dist3;
            z -= G * mi * dz / dist3;
        }
    }
    if (threadIdx.x < n) {
        atomicAdd(&ax[threadIdx.x], x);
        atomicAdd(&ay[threadIdx.x], y);
        atomicAdd(&az[threadIdx.x], z);
        /* ax[blockIdx.x] += x;
        ay[blockIdx.x] += y;
        az[blockIdx.x] += z; */
    }
}
__global__ void problem1Update(double* qx, double* qy, double* qz,
                               double* vx, double* vy, double* vz,
                               double* ax, double* ay, double* az,
                               int planet, int asteroid, double* min_dist) {
    // update positions
    vx[threadIdx.x] += ax[threadIdx.x] * dt;
    vy[threadIdx.x] += ay[threadIdx.x] * dt;
    vz[threadIdx.x] += az[threadIdx.x] * dt;

    qx[threadIdx.x] += vx[threadIdx.x] * dt;
    qy[threadIdx.x] += vy[threadIdx.x] * dt;
    qz[threadIdx.x] += vz[threadIdx.x] * dt;
    __syncthreads();
    if (threadIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        *min_dist = MIN(*min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }
}
__global__ void problem2Update(double* qx, double* qy, double* qz,
                               double* vx, double* vy, double* vz,
                               double* ax, double* ay, double* az,
                               int planet, int asteroid, int* hit_time_step, int step,
                               double* minDis, int* minStep) {
    // update positions
    vx[threadIdx.x] += ax[threadIdx.x] * dt;
    vy[threadIdx.x] += ay[threadIdx.x] * dt;
    vz[threadIdx.x] += az[threadIdx.x] * dt;

    qx[threadIdx.x] += vx[threadIdx.x] * dt;
    qy[threadIdx.x] += vy[threadIdx.x] * dt;
    qz[threadIdx.x] += vz[threadIdx.x] * dt;
    __syncthreads();
    if (threadIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];

        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius)
            *hit_time_step = step;
        if (dx * dx + dy * dy + dz * dz < *minDis) {
            *minStep = step;
            *minDis = dx * dx + dy * dy + dz * dz;
        }
    }
}
__global__ void problem3Update(double* qx, double* qy, double* qz,
                               double* vx, double* vy, double* vz,
                               double* ax, double* ay, double* az, double* m,
                               int planet, int asteroid, int deviceID, int step,
                               bool* isHit, bool* isSave, double* cost, bool isUpdateVec) {
    if (!isUpdateVec) {
        vx[threadIdx.x] += ax[threadIdx.x] * dt;
        vy[threadIdx.x] += ay[threadIdx.x] * dt;
        vz[threadIdx.x] += az[threadIdx.x] * dt;

        qx[threadIdx.x] += vx[threadIdx.x] * dt;
        qy[threadIdx.x] += vy[threadIdx.x] * dt;
        qz[threadIdx.x] += vz[threadIdx.x] * dt;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double dx = qx[planet] - qx[deviceID];
        double dy = qy[planet] - qy[deviceID];
        double dz = qz[planet] - qz[deviceID];
        double dxAsteroid = qx[planet] - qx[asteroid];
        double dyAsteroid = qy[planet] - qy[asteroid];
        double dzAsteroid = qz[planet] - qz[asteroid];
        double dis = step * dt * missile_speed;
        if ((dx * dx + dy * dy + dz * dz) <= (dis * dis) && !*isHit) {
            m[deviceID] = 0;
            *isHit = true;
            *cost = get_missile_cost((step + 1) * dt);
        }
        if (dxAsteroid * dxAsteroid + dyAsteroid * dyAsteroid + dzAsteroid * dzAsteroid < planet_radius * planet_radius) {
            *isSave = false;
        }
    }
}
__global__ void copyVec(double* qx, double* qy, double* qz,
                        double* vx, double* vy, double* vz,
                        double* ax, double* ay, double* az,
                        double* qx_temp, double* qy_temp, double* qz_temp,
                        double* vx_temp, double* vy_temp, double* vz_temp,
                        double* ax_temp, double* ay_temp, double* az_temp) {
    qx[threadIdx.x] = qx_temp[threadIdx.x];
    qy[threadIdx.x] = qy_temp[threadIdx.x];
    qz[threadIdx.x] = qz_temp[threadIdx.x];
    vx[threadIdx.x] = vx_temp[threadIdx.x];
    vy[threadIdx.x] = vy_temp[threadIdx.x];
    vz[threadIdx.x] = vz_temp[threadIdx.x];
    ax[threadIdx.x] = ax_temp[threadIdx.x];
    ay[threadIdx.x] = ay_temp[threadIdx.x];
    az[threadIdx.x] = az_temp[threadIdx.x];
}
void setBody(double* qx, double* qy, double* qz,
             double* vx, double* vy, double* vz, double* m,
             std::vector<double>& temp_qx, std::vector<double>& temp_qy, std::vector<double>& temp_qz,
             std::vector<double>& temp_vx, std::vector<double>& temp_vy, std::vector<double>& temp_vz,
             std::vector<double>& temp_m, std::vector<int>& devicePos) {
    int n = temp_qx.size();
    for (int i = 0; i < (2 + devicePos.size()); i++) {
        cudaMemcpy(&qx[i * n], &temp_qx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&qy[i * n], &temp_qy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&qz[i * n], &temp_qz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&vx[i * n], &temp_vx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&vy[i * n], &temp_vy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&vz[i * n], &temp_vz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(&m[i * n], &temp_m[0], sizeof(double) * n, cudaMemcpyHostToDevice);
    }
}
void typeTrans(std::vector<std::string>& type_temp, std::vector<char>& type, std::vector<int>& devicePos) {
    for (int i = 0; i < type_temp.size(); i++) {
        if (type_temp[i] == "black_hole")
            type.push_back(0);
        else if (type_temp[i] == "star")
            type.push_back(1);
        else if (type_temp[i] == "planet")
            type.push_back(2);
        else if (type_temp[i] == "satellite")
            type.push_back(3);
        else if (type_temp[i] == "asteroid")
            type.push_back(4);
        else if (type_temp[i] == "device") {
            type.push_back(5);
            devicePos.push_back(i);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3)
        throw std::runtime_error("must supply 2 arguments");
    int n, planet, asteroid, gpuNum;
    cudaGetDeviceCount(&gpuNum);
    // std::vector<double> qx, qy, qz, vx, vy, vz, m;
    double *qx, *qy, *qz, *vx, *vy, *vz, *ax, *ay, *az, *m;
    double *qx_second, *qy_second, *qz_second, *vx_second, *vy_second, *vz_second, *ax_second, *ay_second, *az_second, *m_second;
    std::vector<double> temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m;
    std::vector<std::string> temp_type;
    std::vector<int> devicePos;
    std::vector<char> type;
    char *typeDevice, *typeDevice_second;
    // type 0=black_hole 1=star 2=planet 3=satellite 4=asteroid 5=device
    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };
    read_input(argv[1], n, planet, asteroid, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m, temp_type);
    typeTrans(temp_type, type, devicePos);
    printf("GPU num: %d\n", gpuNum);
    if (n < 45)
        gpuNum = 1;
    if (gpuNum < 2) {
        cudaMalloc(&qx, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&qy, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&qz, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&vx, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&vy, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&vz, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&ax, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&ay, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&az, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&m, sizeof(double) * n * (2 + devicePos.size()));
        cudaMalloc(&typeDevice, sizeof(char) * n * 2);
        cudaMemcpy(typeDevice, &type[0], sizeof(char) * type.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(&typeDevice[n], &type[0], sizeof(char) * type.size(), cudaMemcpyHostToDevice);
        setBody(qx, qy, qz, vx, vy, vz, m, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m, devicePos);
    } else {
        cudaSetDevice(0);
        cudaMalloc(&qx, sizeof(double) * n);
        cudaMalloc(&qy, sizeof(double) * n);
        cudaMalloc(&qz, sizeof(double) * n);
        cudaMalloc(&vx, sizeof(double) * n);
        cudaMalloc(&vy, sizeof(double) * n);
        cudaMalloc(&vz, sizeof(double) * n);
        cudaMalloc(&ax, sizeof(double) * n);
        cudaMalloc(&ay, sizeof(double) * n);
        cudaMalloc(&az, sizeof(double) * n);
        cudaMalloc(&m, sizeof(double) * n);
        cudaMalloc(&typeDevice, sizeof(char) * n);
        cudaMemcpy(typeDevice, &type[0], sizeof(char) * type.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(qx, &temp_qx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(qy, &temp_qy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(qz, &temp_qz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(vx, &temp_vx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(vy, &temp_vy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(vz, &temp_vz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(m, &temp_m[0], sizeof(double) * n, cudaMemcpyHostToDevice);

        cudaSetDevice(1);
        cudaMalloc(&qx_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&qy_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&qz_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&vx_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&vy_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&vz_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&ax_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&ay_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&az_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&m_second, sizeof(double) * n * (1 + devicePos.size()));
        cudaMalloc(&typeDevice_second, sizeof(char) * n);
        cudaMemcpy(typeDevice_second, &type[0], sizeof(char) * type.size(), cudaMemcpyHostToDevice);
        for (int i = 0; i < (1 + devicePos.size()); i++) {
            cudaMemcpy(&qx_second[i * n], &temp_qx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&qy_second[i * n], &temp_qy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&qz_second[i * n], &temp_qz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&vx_second[i * n], &temp_vx[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&vy_second[i * n], &temp_vy[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&vz_second[i * n], &temp_vz[0], sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(&m_second[i * n], &temp_m[0], sizeof(double) * n, cudaMemcpyHostToDevice);
        }
    }
    //  Problem 1
    cudaStream_t stream[6];
    if (gpuNum < 2)
        for (int i = 0; i < 6; i++)
            cudaStreamCreate(&stream[i]);
    else {
        cudaSetDevice(0);
        cudaStreamCreate(&stream[0]);
        cudaSetDevice(1);
        for (int i = 1; i < 6; i++)
            cudaStreamCreate(&stream[i]);
    }

    double min_dist = std::numeric_limits<double>::infinity();
    double* min_dist_device;
    int hit_time_step = -2;
    int* hit_time_step_device;
    int gravity_device_id = -1;
    int* gravity_device_id_device;
    double missile_cost = std::numeric_limits<double>::infinity();

    bool isHit[4] = {false, false, false, false};
    bool isSave[4] = {true, true, true, true};
    double cost[4] = {0, 0, 0, 0};
    bool* isHit_device;
    bool* isSave_device;
    double* cost_device;
    double minDis = std::numeric_limits<double>::infinity();
    double* minDis_device;
    int minStep;
    int* minStep_device;

    cudaSetDevice(0);
    cudaMalloc(&min_dist_device, sizeof(double));
    cudaMemcpy(min_dist_device, &min_dist, sizeof(double), cudaMemcpyHostToDevice);
    if (gpuNum >= 2)
        cudaSetDevice(1);
    cudaMalloc(&gravity_device_id_device, sizeof(int));
    cudaMalloc(&hit_time_step_device, sizeof(int));
    cudaMalloc(&minDis_device, sizeof(double));
    cudaMalloc(&minStep_device, sizeof(int));
    cudaMalloc(&isHit_device, sizeof(bool) * 4);
    cudaMalloc(&isSave_device, sizeof(bool) * 4);
    cudaMalloc(&cost_device, sizeof(double) * 4);
    cudaMemcpy(hit_time_step_device, &hit_time_step, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gravity_device_id_device, &gravity_device_id, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(isHit_device, isHit, sizeof(bool) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(isSave_device, isSave, sizeof(bool) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(minDis_device, &minDis, sizeof(double), cudaMemcpyHostToDevice);

    int thread = (n >= 60) ? 1024 : 64;
    int block = (n >= 60) ? 128 : 64;
    if (n >= 512) block = 256;
    for (int step = 0; step <= n_steps; step++) {
        cudaSetDevice(0);
        run_step<<<block, thread, 0, stream[0]>>>(false, step, n, qx, qy, qz, ax, ay, az, m, typeDevice);
        if (gpuNum >= 2)
            cudaSetDevice(1);
        if (gpuNum >= 2)
            run_step<<<block, thread, 0, stream[1]>>>(true, step, n, qx_second, qy_second, qz_second, ax_second, ay_second, az_second, m_second, typeDevice_second);
        else
            run_step<<<block, thread, 0, stream[1]>>>(true, step, n, &qx[n], &qy[n], &qz[n], &ax[n], &ay[n], &az[n], &m[n], &typeDevice[n]);
        for (int i = 2; i < (2 + devicePos.size()); i++)
            if (isSave[i - 2] && isHit[i - 2]) {
                if (gpuNum >= 2)
                    run_step<<<block, thread, 0, stream[i]>>>(true, step, n, &qx_second[(i - 1) * n], &qy_second[(i - 1) * n], &qz_second[(i - 1) * n], &ax_second[(i - 1) * n], &ay_second[(i - 1) * n], &az_second[(i - 1) * n], &m_second[(i - 1) * n], typeDevice_second);
                else
                    run_step<<<block, thread, 0, stream[i]>>>(true, step, n, &qx[i * n], &qy[i * n], &qz[i * n], &ax[i * n], &ay[i * n], &az[i * n], &m[i * n], &typeDevice[n]);
            }
        for (int i = 0; i < (2 + devicePos.size()); i++)
            cudaStreamSynchronize(stream[i]);
        cudaSetDevice(0);
        problem1Update<<<1, n, 0, stream[0]>>>(qx, qy, qz, vx, vy, vz, ax, ay, az, planet, asteroid, min_dist_device);
        if (gpuNum >= 2)
            cudaSetDevice(1);
        if (hit_time_step == -2) {
            if (gpuNum >= 2)
                problem2Update<<<1, n, 0, stream[1]>>>(qx_second, qy_second, qz_second, vx_second, vy_second, vz_second, ax_second, ay_second, az_second, planet, asteroid, hit_time_step_device, step, minDis_device, minStep_device);
            else
                problem2Update<<<1, n, 0, stream[1]>>>(&qx[n], &qy[n], &qz[n], &vx[n], &vy[n], &vz[n], &ax[n], &ay[n], &az[n], planet, asteroid, hit_time_step_device, step, minDis_device, minStep_device);
        }

        for (int i = 2; i < (2 + devicePos.size()); i++)
            if (!isHit[i - 2]) {
                if (gpuNum >= 2)
                    copyVec<<<1, n, 0, stream[1]>>>(&qx_second[(i - 1) * n], &qy_second[(i - 1) * n], &qz_second[(i - 1) * n], &vx_second[(i - 1) * n], &vy_second[(i - 1) * n], &vz_second[(i - 1) * n], &ax_second[(i - 1) * n], &ay_second[(i - 1) * n], &az_second[(i - 1) * n], qx_second, qy_second, qz_second, vx_second, vy_second, vz_second, ax_second, ay_second, az_second);
                else
                    copyVec<<<1, n, 0, stream[1]>>>(&qx[i * n], &qy[i * n], &qz[i * n], &vx[i * n], &vy[i * n], &vz[i * n], &ax[i * n], &ay[i * n], &az[i * n], &qx[n], &qy[n], &qz[n], &vx[n], &vy[n], &vz[n], &ax[n], &ay[n], &az[n]);
            }

        for (int i = 2; i < (2 + devicePos.size()); i++)
            if (isSave[i - 2]) {
                if (gpuNum >= 2) {
                    if (!isHit[i - 2])
                        problem3Update<<<1, n, 0, stream[1]>>>(&qx_second[(i - 1) * n], &qy_second[(i - 1) * n], &qz_second[(i - 1) * n], &vx_second[(i - 1) * n], &vy_second[(i - 1) * n], &vz_second[(i - 1) * n], &ax_second[(i - 1) * n], &ay_second[(i - 1) * n], &az_second[(i - 1) * n], &m_second[(i - 1) * n], planet, asteroid, devicePos[i - 2], step, &isHit_device[i - 2], &isSave_device[i - 2], &cost_device[i - 2], true);
                    else
                        problem3Update<<<1, n, 0, stream[i]>>>(&qx_second[(i - 1) * n], &qy_second[(i - 1) * n], &qz_second[(i - 1) * n], &vx_second[(i - 1) * n], &vy_second[(i - 1) * n], &vz_second[(i - 1) * n], &ax_second[(i - 1) * n], &ay_second[(i - 1) * n], &az_second[(i - 1) * n], &m_second[(i - 1) * n], planet, asteroid, devicePos[i - 2], step, &isHit_device[i - 2], &isSave_device[i - 2], &cost_device[i - 2], false);
                } else {
                    if (!isHit[i - 2])
                        problem3Update<<<1, n, 0, stream[1]>>>(&qx[i * n], &qy[i * n], &qz[i * n], &vx[i * n], &vy[i * n], &vz[i * n], &ax[i * n], &ay[i * n], &az[i * n], &m[i * n], planet, asteroid, devicePos[i - 2], step, &isHit_device[i - 2], &isSave_device[i - 2], &cost_device[i - 2], true);
                    else
                        problem3Update<<<1, n, 0, stream[i]>>>(&qx[i * n], &qy[i * n], &qz[i * n], &vx[i * n], &vy[i * n], &vz[i * n], &ax[i * n], &ay[i * n], &az[i * n], &m[i * n], planet, asteroid, devicePos[i - 2], step, &isHit_device[i - 2], &isSave_device[i - 2], &cost_device[i - 2], false);
                }
            }
        for (int i = 0; i < (2 + devicePos.size()); i++)
            cudaStreamSynchronize(stream[i]);
        cudaMemcpy(&hit_time_step, hit_time_step_device, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(isSave, isSave_device, sizeof(bool) * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(isHit, isHit_device, sizeof(bool) * 4, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaMemcpy(&min_dist, min_dist_device, sizeof(double), cudaMemcpyDeviceToHost);
    if (gpuNum >= 2)
        cudaSetDevice(1);
    cudaMemcpy(&minStep, minStep_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cost, cost_device, sizeof(double) * 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < devicePos.size(); i++) {
        if (isSave[i] && cost[i] < missile_cost) {
            missile_cost = cost[i];
            gravity_device_id = devicePos[i];
        }
    }
    if (hit_time_step == -2)
        hit_time_step = minStep;
    if (gravity_device_id == -1)
        missile_cost = 0;
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    cudaDeviceReset();
}
