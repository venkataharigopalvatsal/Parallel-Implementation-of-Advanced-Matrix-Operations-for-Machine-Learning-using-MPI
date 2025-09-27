#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int A[N][N], B[N][N], C[N][N];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Matrix size N must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;
    
    // Local buffers sized dynamically
    int local_A[rows_per_proc][N];
    int local_B[rows_per_proc][N];
    int local_C[rows_per_proc][N];

    if (rank == 0) {
        printf("Initializing matrices A and B...\n");
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j + 1; // example data
                B[i][j] = 2;         // example data
            }
    }

    // Scatter rows of A and B to local buffers
    MPI_Scatter(A, rows_per_proc * N, MPI_INT, local_A, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, rows_per_proc * N, MPI_INT, local_B, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute Hadamard product locally
    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++)
            local_C[i][j] = local_A[i][j] * local_B[i][j];

    // Gather the results back to C on root
    MPI_Gather(local_C, rows_per_proc * N, MPI_INT, C, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nMatrix A:\n");
        print_matrix(A);
        printf("\nMatrix B:\n");
        print_matrix(B);
        printf("\nHadamard Product Matrix C:\n");
        print_matrix(C);
    }

    MPI_Finalize();
    return 0;
}
