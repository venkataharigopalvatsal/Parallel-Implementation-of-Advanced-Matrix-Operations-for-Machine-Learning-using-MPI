#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "mpi.h"

#define N 64    // Matrix size
bool printResults = true;

// Prints an NxN matrix
void printMatrix(int *matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++)
            printf("%d\t", matrix[i*cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int numProcs, rank;
    int *matrix1 = NULL, *matrix2 = NULL, *productMatrix = NULL;
    int *subMatrix1 = NULL, *subProduct = NULL;
    int rows, extra, offset = 0;
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int numWorkers = numProcs - 1;

    // Dynamically allocate to allow larger matrices and reduce stack usage
    if(rank == 0) {
        matrix1 = malloc(N * N * sizeof(int));
        matrix2 = malloc(N * N * sizeof(int));
        productMatrix = malloc(N * N * sizeof(int));

        srand(time(NULL));
        for(int i = 0; i < N*N; i++) {
            matrix1[i] = rand() % 6 + 1;
            matrix2[i] = rand() % 6 + 1;
        }

        startTime = MPI_Wtime();

        // Distribute work to workers; handle leftovers if N%numWorkers != 0
        rows = N / numWorkers;
        extra = N % numWorkers;
        offset = 0;
        for(int dest=1; dest <= numWorkers; dest++) {
            int myRows = rows + (dest <= extra ? 1 : 0);
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&myRows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix1[offset*N], myRows*N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(matrix2, N*N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            offset += myRows;
        }
        // Gather results
        for(int src=1; src <= numWorkers; src++) {
            int rcvOffset, rcvRows;
            MPI_Recv(&rcvOffset, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rcvRows, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&productMatrix[rcvOffset*N], rcvRows*N, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        endTime = MPI_Wtime();
        if(printResults) {
            printf("Matrix 1:\n");
            printMatrix(matrix1, N, N);
            printf("Matrix 2:\n");
            printMatrix(matrix2, N, N);
            printf("Product Matrix:\n");
            printMatrix(productMatrix, N, N);
        }
        printf("Runtime: %lf seconds\n", endTime - startTime);

        free(matrix1); free(matrix2); free(productMatrix);
    } else {
        int start_row, myRows;
        MPI_Recv(&start_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&myRows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        subMatrix1 = malloc(myRows * N * sizeof(int));
        matrix2 = malloc(N * N * sizeof(int));
        subProduct = malloc(myRows * N * sizeof(int));

        MPI_Recv(subMatrix1, myRows*N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(matrix2, N*N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Matrix multiplication
        for(int i=0; i<myRows; i++) {
            for(int j=0; j<N; j++) {
                int sum = 0;
                for(int k=0; k<N; k++)
                    sum += subMatrix1[i*N + k] * matrix2[k*N + j];
                subProduct[i*N + j] = sum;
            }
        }
        MPI_Send(&start_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&myRows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(subProduct, myRows*N, MPI_INT, 0, 2, MPI_COMM_WORLD);

        free(subMatrix1); free(matrix2); free(subProduct);
    }
    MPI_Finalize();
    return 0;
}
