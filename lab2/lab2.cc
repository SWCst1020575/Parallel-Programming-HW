#include <math.h>
#include <mpi.h>

#include <iostream>
#include <string>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    unsigned long long int r(std::atoll(argv[1]));
    unsigned long long int k(std::atoll(argv[2]));
    // Get world size and rank
    int rank, world_size;
    unsigned long long int sum = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    unsigned long long int end = ((r / world_size + 1) * (rank + 1) < r) ? (r / world_size + 1) * (rank + 1) : r;

    for (unsigned long long int i = rank * (r / world_size + 1); i < end; i++) {
        unsigned long long int rSquare = r * r;
        unsigned long long int iSquare = i * i;
        unsigned long long int predict = floor(sqrt(rSquare - iSquare));
        while (predict * predict + iSquare < rSquare)
            predict++;
        sum += predict;
    }
    sum %= k;
    unsigned long long int total;

    MPI_Reduce(&sum, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << (total * (unsigned long long int)4) % k << '\n';
    MPI_Finalize();

    return 0;
}
