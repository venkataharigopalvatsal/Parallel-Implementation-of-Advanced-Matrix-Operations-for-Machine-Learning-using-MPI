#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "mpi.h"

#define N 64
bool printResults = true;

void printMatrix(int matrix[N][N]);

int main(int argc, char **argv) {
    int matrix1[N][N], matrix2[N][N], productMatrix[N][N];
    int i, j, k;

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_workers = num_procs - 1;

    if (rank == 0) {
        // Initialize matrices
        srand(time(NULL));
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                matrix1[i][j] = (rand() % 6) + 1;
                matrix2[i][j] = (rand() % 6) + 1;
            }

        int num_rows = N / num_workers;
        int offset = 0;

        clock_t begin = clock();
        printf("Multiplying %dx%d matrix using %d processors...\n\n", N, N, num_procs);

        // Send chunks of A and full B to workers
        for (int dest = 1; dest <= num_workers; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&num_rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix1[offset][0], num_rows * N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix2[0][0], N * N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            offset += num_rows;
        }

        // Receive results from workers
        for (int source = 1; source <= num_workers; source++) {
            int recv_offset, rows;
            MPI_Recv(&recv_offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&productMatrix[recv_offset][0], rows * N, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        clock_t end = clock();
        if (printResults) {
            printf("Matrix 1:\n"); printMatrix(matrix1);
            printf("Matrix 2:\n"); printMatrix(matrix2);
            printf("Product Matrix:\n"); printMatrix(productMatrix);
        }
        printf("Runtime: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
    } else {
        // Worker processes
        int offset, num_rows;
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&num_rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int subMatrix[num_rows][N];
        MPI_Recv(&subMatrix, num_rows * N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int matrix2_full[N][N];
        MPI_Recv(&matrix2_full, N * N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int subProduct[num_rows][N];
        for (i = 0; i < num_rows; i++)
            for (j = 0; j < N; j++) {
                subProduct[i][j] = 0;
                for (k = 0; k < N; k++)
                    subProduct[i][j] += subMatrix[i][k] * matrix2_full[k][j];
            }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&num_rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&subProduct, num_rows * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

void printMatrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d\t", matrix[i][j]);
        printf("\n");
    }
    printf("\n");
}
