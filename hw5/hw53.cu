#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define BLOCK 8
#define THREAD 1024
#define SIZE 1024

__constant__ const int n_steps = 200000;
__constant__ const double dt = 60;
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

__device__ void run_step(int step, int n, int i, int j,
                         double* qx, double* qy, double* qz,
                         double* vx, double* vy, double* vz,
                         double* ax, double* ay, double* az,
                         const double* m, const char* type) {
    // compute accelerations

    if (j == i) return;
    double mj = m[j];
    if (type[j] == 5 && mj > 0)
        mj = gravity_device_mass(mj, step * dt);
    double dx = qx[j] - qx[i];
    double dy = qy[j] - qy[i];
    double dz = qz[j] - qz[i];
    double dist3 =
        pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);
    ax[i] += G * mj * dx / dist3;
    ay[i] += G * mj * dy / dist3;
    az[i] += G * mj * dz / dist3;
}
__global__ void problem1(int curStep, int n, int planet, int asteroid, double* min_dist,
                         double* qx, double* qy, double* qz,
                         double* vx, double* vy, double* vz,
                         double* m, char* type) {
    __shared__ double ax[THREAD];
    __shared__ double ay[THREAD];
    __shared__ double az[THREAD];
    int step = curStep + blockIdx.x;
    int i = threadIdx.x;
    if (i < n) {
        ax[i] = 0;
        ay[i] = 0;
        az[i] = 0;
        for (int j = 0; j < n; j++)
            run_step(step, n, i, j, qx, qy, qz, vx, vy, vz, ax, ay, az, m, type);
        // update velocities
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += az[i] * dt;

        // update positions
        qx[i] += vx[i] * dt;
        qy[i] += vy[i] * dt;
        qz[i] += vz[i] * dt;
        __syncthreads();
        // run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        if (i == 0) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            *min_dist = MIN(*min_dist, sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
}
void setBody(double* qx, double* qy, double* qz,
             double* vx, double* vy, double* vz, double* m,
             std::vector<double>& temp_qx, std::vector<double>& temp_qy, std::vector<double>& temp_qz,
             std::vector<double>& temp_vx, std::vector<double>& temp_vy, std::vector<double>& temp_vz, std::vector<double>& temp_m) {
    cudaMemcpy(qx, &temp_qx[0], sizeof(double) * temp_qx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(qy, &temp_qy[0], sizeof(double) * temp_qy.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(qz, &temp_qz[0], sizeof(double) * temp_qz.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vx, &temp_vx[0], sizeof(double) * temp_vx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vy, &temp_vy[0], sizeof(double) * temp_vy.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vz, &temp_vz[0], sizeof(double) * temp_vz.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m, &temp_m[0], sizeof(double) * temp_m.size(), cudaMemcpyHostToDevice);
}
void typeTrans(std::vector<std::string>& type_temp, std::vector<char>& type) {
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
        else if (type_temp[i] == "device")
            type.push_back(5);
    }
}
int main(int argc, char** argv) {
    if (argc != 3)
        throw std::runtime_error("must supply 2 arguments");
    int n, planet, asteroid;
    // std::vector<double> qx, qy, qz, vx, vy, vz, m;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    std::vector<double> temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m;
    std::vector<std::string> temp_type;
    std::vector<char> type;
    char* typeDevice;
    // type 0=black_hole 1=star 2=planet 3=satellite 4=asteroid 5=device
    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };
    read_input(argv[1], n, planet, asteroid, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m, temp_type);
    typeTrans(temp_type, type);
    cudaMalloc(&qx, sizeof(double) * n);
    cudaMalloc(&qy, sizeof(double) * n);
    cudaMalloc(&qz, sizeof(double) * n);
    cudaMalloc(&vx, sizeof(double) * n);
    cudaMalloc(&vy, sizeof(double) * n);
    cudaMalloc(&vz, sizeof(double) * n);
    cudaMalloc(&m, sizeof(double) * n);
    cudaMalloc(&typeDevice, sizeof(char) * n);
    cudaMemcpy(typeDevice, &type[0], sizeof(char) * type.size(), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++)
        if (type[i] == 5)
            temp_m[i] = 0;
    setBody(qx, qy, qz, vx, vy, vz, m, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m);
    //  Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    double* min_dist_device;
    cudaMalloc(&min_dist_device, sizeof(double));
    cudaMemcpy(min_dist_device, &min_dist, sizeof(double), cudaMemcpyHostToDevice);
    int hit_time_step = -2;
    for (int step = 0; step <= n_steps; step += BLOCK) {
        problem1<<<BLOCK, THREAD>>>(step, n, planet, asteroid, min_dist_device, qx, qy, qz, vx, vy, vz, m, typeDevice);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&min_dist, min_dist_device, sizeof(double), cudaMemcpyDeviceToHost);
    // Problem 2

    /* resetBody(qx, qy, qz, vx, vy, vz, m, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m);
    for (int step = 0; step <= n_steps; step++) {
        run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius) {
            hit_time_step = step;
            break;
        }
    } */

    // Problem 3
    // TODO
    int gravity_device_id = -1;
    double missile_cost = std::numeric_limits<double>::infinity();
    /*
        for (int i = 0; i < n; i++) {
            if (type[i] != 5)
                continue;
            resetBody(qx, qy, qz, vx, vy, vz, m, temp_qx, temp_qy, temp_qz, temp_vx, temp_vy, temp_vz, temp_m);
            bool isHit = false;
            bool isSave = true;
            double cost;
            for (int step = 0; step <= n_steps; step++) {
                run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);

                double dx = qx[planet] - qx[i];
                double dy = qy[planet] - qy[i];
                double dz = qz[planet] - qz[i];
                double dxAsteroid = qx[planet] - qx[asteroid];
                double dyAsteroid = qy[planet] - qy[asteroid];
                double dzAsteroid = qz[planet] - qz[asteroid];
                double dis = step * dt * missile_speed;
                if ((dx * dx + dy * dy + dz * dz) <= (dis * dis) && !isHit) {
                    m[i] = 0;
                    isHit = true;
                    cost = get_missile_cost((step + 1) * dt);
                }
                if (dxAsteroid * dxAsteroid + dyAsteroid * dyAsteroid + dzAsteroid * dzAsteroid < planet_radius * planet_radius) {
                    isSave = false;
                    break;
                }
            }
            if (isSave && cost < missile_cost) {
                missile_cost = cost;
                gravity_device_id = i;
            }
        } */
    if (gravity_device_id == -1)
        missile_cost = 0;
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    cudaDeviceReset();
}
