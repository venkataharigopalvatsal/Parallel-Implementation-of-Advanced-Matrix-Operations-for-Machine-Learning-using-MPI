#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

// --- Utility functions ---
double** allocate_matrix(int rows, int cols) {
    double* data = (double*)calloc(rows * cols, sizeof(double));
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        array[i] = &(data[i * cols]);
    return array;
}

void print_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void free_matrix(double** matrix) {
    free(matrix[0]);
    free(matrix);
}

void multiply_add(double** A, double** B, double** C, int m, int n, int k) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int l = 0; l < k; l++)
                C[i][j] += A[i][l] * B[l][j];
}

void add_matrix(double** A, double** B, double** C, int m, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract_matrix(double** A, double** B, double** C, int m, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void hadamard(double** A, double** B, double** C, int m, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] * B[i][j];
}

// --- Gradient Descent ---
void gradient_descent(double** X, double** Y, double** W, int m, int n, int output_dim, int iterations, double alpha, int num_procs) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double** pred = allocate_matrix(m, output_dim);
    double** error = allocate_matrix(m, output_dim);
    double** grad_local = allocate_matrix(n, output_dim);
    double* grad_global = (double*)calloc(n * output_dim, sizeof(double));
    double** X_T = allocate_matrix(n, m);

    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            X_T[j][i] = X[i][j];

    for (int it = 0; it < iterations; it++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < output_dim; j++)
                grad_local[i][j] = 0.0;

        multiply_add(X, W, pred, m, output_dim, n);
        subtract_matrix(pred, Y, error, m, output_dim);
        multiply_add(X_T, error, grad_local, n, output_dim, m);

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < output_dim; j++)
                grad_global[i * output_dim + j] = grad_local[i][j];

        double* grad_total = (double*)calloc(n * output_dim, sizeof(double));
        MPI_Allreduce(grad_global, grad_total, n * output_dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < output_dim; j++)
                W[i][j] -= alpha * grad_total[i * output_dim + j] / (m * num_procs);

        free(grad_total);
    }

    free_matrix(pred);
    free_matrix(error);
    free_matrix(grad_local);
    free_matrix(X_T);
    free(grad_global);
}

// --- Main ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np <num_procs> %s <matrix_size> <operation>\nOperations: a=add, s=subtract, m=multiply, h=hadamard, g=gradient_descent\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    char op = argv[2][0];

    int dims[2] = {0, 0};
    MPI_Dims_create(num_procs, 2, dims);
    int p_rows = dims[0], p_cols = dims[1];

    if (N % p_rows != 0 || N % p_cols != 0) {
        if (rank == 0) fprintf(stderr, "Matrix size N=%d must be divisible by process grid %d x %d\n", N, p_rows, p_cols);
        MPI_Finalize();
        return 1;
    }

    int periods[2] = {1, 1}; // Use periodic boundaries for easy shifting
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);

    int coords[2];
    MPI_Cart_coords(comm_cart, rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    int local_rows = N / p_rows;
    int local_cols = N / p_cols;

    double** local_A = allocate_matrix(local_rows, local_cols);
    double** local_B = allocate_matrix(local_rows, local_cols);
    double** local_C = allocate_matrix(local_rows, local_cols);

    srand(rank + 1);
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < local_cols; j++) {
            local_A[i][j] = ((double)(rand() % 10)) / 9.0;
            local_B[i][j] = ((double)(rand() % 10)) / 9.0;
        }

    MPI_Barrier(comm_cart);
    double start_time = MPI_Wtime();

    switch (op) {
        case 'm': { // Cannon's Algorithm
            if (p_rows != p_cols) {
                if (rank == 0) fprintf(stderr, "Cannon's algorithm requires a square process grid (p_rows == p_cols).\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Initial alignment of matrices
            int left, right, up, down;
            MPI_Cart_shift(comm_cart, 1, -my_row, &right, &left);
            MPI_Sendrecv_replace(local_A[0], local_rows * local_cols, MPI_DOUBLE, left, 1, right, 1, comm_cart, MPI_STATUS_IGNORE);

            MPI_Cart_shift(comm_cart, 0, -my_col, &down, &up);
            MPI_Sendrecv_replace(local_B[0], local_rows * local_cols, MPI_DOUBLE, up, 1, down, 1, comm_cart, MPI_STATUS_IGNORE);
            
            // Main computation loop
            for (int i = 0; i < p_rows; i++) {
                multiply_add(local_A, local_B, local_C, local_rows, local_cols, local_cols);

                // Shift A left by 1
                MPI_Cart_shift(comm_cart, 1, -1, &right, &left);
                MPI_Sendrecv_replace(local_A[0], local_rows * local_cols, MPI_DOUBLE, left, 1, right, 1, comm_cart, MPI_STATUS_IGNORE);

                // Shift B up by 1
                MPI_Cart_shift(comm_cart, 0, -1, &down, &up);
                MPI_Sendrecv_replace(local_B[0], local_rows * local_cols, MPI_DOUBLE, up, 1, down, 1, comm_cart, MPI_STATUS_IGNORE);
            }
            break;
        }
        case 'a': add_matrix(local_A, local_B, local_C, local_rows, local_cols); break;
        case 's': subtract_matrix(local_A, local_B, local_C, local_rows, local_cols); break;
        case 'h': hadamard(local_A, local_B, local_C, local_rows, local_cols); break;
        case 'g': {
            // ... (gradient descent code remains the same)
            break;
        }
        default:
            if (rank == 0) fprintf(stderr, "Unknown operation %c\n", op);
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // For verification: if we are running on a single process, print the result matrix to stdout.
    if (num_procs == 1) {
        if (op == 'm' || op == 'a' || op == 's' || op == 'h') {
             print_matrix(local_C, local_rows, local_cols);
        }
    }

    MPI_Barrier(comm_cart);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        // Print ALL summary info to stderr. This prevents it from interfering
        // with the stdout capture for the correctness check.
        fprintf(stderr, "Hybrid (MPI+OpenMP) Operation: %c\n", op);
        fprintf(stderr, "Matrix Size: %d x %d\n", N, N);
        fprintf(stderr, "Processes: %d (%d x %d grid)\n", num_procs, p_rows, p_cols);
        fprintf(stderr, "Time Elapsed: %f seconds\n", end_time - start_time);
    }

    free_matrix(local_A);
    free_matrix(local_B);
    free_matrix(local_C);

    MPI_Comm_free(&comm_cart);
    MPI_Finalize();
    return 0;
}
