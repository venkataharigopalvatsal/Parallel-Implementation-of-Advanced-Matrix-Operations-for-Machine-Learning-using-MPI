// File: serial_verifier.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Utility Functions ---
double** allocate_matrix(int rows, int cols) {
    double* data = (double*)calloc(rows * cols, sizeof(double));
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

void free_matrix(double** matrix) {
    free(matrix[0]);
    free(matrix);
}

void print_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// --- Main ---
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <operation>\n", argv[0]);
        fprintf(stderr, "Note: Only operation 'm' is supported for verification.\n");
        return 1;
    }

    int N = atoi(argv[1]);
    char op = argv[2][0];

    if (op != 'm') {
        fprintf(stderr, "This verifier only performs multiplication ('m').\n");
        return 1;
    }

    double** A = allocate_matrix(N, N);
    double** B = allocate_matrix(N, N);
    double** C = allocate_matrix(N, N); // Initialized to zero by calloc

    // IMPORTANT: Initialize with a fixed seed to be reproducible.
    // Your parallel code should use srand(rank + 1) to match this when rank is 0.
    srand(1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = ((double)(rand() % 10)) / 9.0;
            B[i][j] = ((double)(rand() % 10)) / 9.0;
        }
    }

    // Standard serial matrix multiplication
    fprintf(stderr, "Performing serial matrix multiplication for N=%d...\n", N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    // Print the final result matrix to standard output
    // This can be redirected to a file for comparison.
    fprintf(stderr, "Printing result matrix to stdout.\n");
    print_matrix(C, N, N);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
