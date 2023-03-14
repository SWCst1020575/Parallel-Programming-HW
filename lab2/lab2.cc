#include <mpi.h>

#include <iostream>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    // Get world size and rank
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::cout << "rank: " << rank << "\n";

    std::cout << "world size: " << world_size << "\n";
    MPI_Finalize();
    return 0;
}
