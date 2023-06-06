#include <iostream>
#include <random>
#define N 1024

#define MIN(x, y) (x) < (y) ? (x) : (y)

double CPU_reduction(double *arr, int n) {
    double ret = arr[0];
    for (int i = 1; i < n; i++) {
        ret = min(ret, arr[i]);
    }
    return ret;
}
void generate_random_doubles(double *arr, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long int> dist;
    for (int i = 0; i < n; ++i) {
        auto c = dist(gen);
        memcpy(&arr[i], &c, sizeof(double));
    }
}
__global__ void cuda_reduction(double *arr, int n, double *ret) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double temp[N];
    if(id < N)
        temp[threadIdx.x] = arr[id];
    
    __syncthreads();
    // atomicMin(ret, arr[threadIdx.x]);
    for(int i = 1; i < blockDim.x; i*=2){
        int index = 2 * i * threadIdx.x;
        if (index < blockDim.x) 
            temp[index] = MIN(temp[index],temp[index + i]);
        __syncthreads();
    }
        
    
    if(threadIdx.x == 0)
        ret[0] = temp[0];
    
}
int main() {
    double *ret = new double;
    double *arr = new double[N];

    double *arrDevice, *retDevice;

    generate_random_doubles(arr, N);
    std::cout << "Generated numbers:";
    for (int i = 0; i < N; i++) {
        std::cout << ' ' << arr[i];
    }
    std::cout << '\n';

    // cudaMalloc and cudaMemcpy is required
    cudaMalloc(&arrDevice, sizeof(double) * N);
    cudaMalloc(&retDevice, sizeof(double));
    cudaMemcpy(arrDevice, arr, sizeof(double) * N, cudaMemcpyHostToDevice);
    cuda_reduction<<<1, N>>>(arrDevice, N, retDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, retDevice, sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "The minimum value: " << *ret << '\n';

    delete ret;
    delete[] arr;
    cudaDeviceReset();
    return 0;
}