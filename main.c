#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matrix.h"

int main() {
    // Define the matrix sizes for testing
    int test_sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // Set number of threads for OpenMP
    omp_set_num_threads(8);
    int num_threads = 8;

    printf("Comparing Parallel Strassen vs. Serial Strassen,\n");
    printf("and Parallel Standard vs. Serial Standard.\n");
    printf("Number of threads: %d\n\n", num_threads);

    for (int idx = 0; idx < num_sizes; idx++) {
        int n = test_sizes[idx];
        printf("===== Matrix Size: %dx%d =====\n", n, n);

        // Allocate matrices and result arrays
        double *A = (double*)malloc(n * n * sizeof(double));
        double *B = (double*)malloc(n * n * sizeof(double));
        double *C_strassen_parallel = (double*)malloc(n * n * sizeof(double));
        double *C_strassen_serial   = (double*)malloc(n * n * sizeof(double));
        double *C_standard_parallel = (double*)malloc(n * n * sizeof(double));
        double *C_standard_serial   = (double*)malloc(n * n * sizeof(double));

        // Initialize A and B with random doubles in [0,1]
        for (int i = 0; i < n*n; i++){
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }

        // --- 1) Parallel Strassen ---
        double start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                strassen_parallel(A, B, C_strassen_parallel, n);
            }
        }
        double end_time = omp_get_wtime();
        double time_strassen_parallel = end_time - start_time;
        printf("Parallel Strassen time: %.4f seconds\n", time_strassen_parallel);

        // --- 2) Serial Strassen ---
        start_time = omp_get_wtime();
        strassen_serial(A, B, C_strassen_serial, n);
        end_time = omp_get_wtime();
        double time_strassen_serial = end_time - start_time;
        printf("Serial Strassen time:   %.4f seconds\n", time_strassen_serial);

        // --- 3) Parallel Standard Multiplication ---
        start_time = omp_get_wtime();
        multiply_standard_parallel(A, B, C_standard_parallel, n);
        end_time = omp_get_wtime();
        double time_standard_parallel = end_time - start_time;
        printf("Parallel Standard time: %.4f seconds\n", time_standard_parallel);

        // --- 4) Serial Standard Multiplication ---
        start_time = omp_get_wtime();
        multiply_standard_serial(A, B, C_standard_serial, n);
        end_time = omp_get_wtime();
        double time_standard_serial = end_time - start_time;
        printf("Serial Standard time:   %.4f seconds\n", time_standard_serial);

        // --- Compare correctness ---
        int pass_strassen = compare_matrices(C_strassen_parallel, C_strassen_serial, n);
        int pass_standard = compare_matrices(C_standard_parallel, C_standard_serial, n);
        int pass_strassen_standard = compare_matrices(C_strassen_parallel, C_standard_serial, n);
        printf("Parallel vs. Serial Strassen compare:   %s\n", pass_strassen ? "PASS" : "FAIL");
        printf("Parallel vs. Serial Standard compare:   %s\n", pass_standard ? "PASS" : "FAIL");
        printf("Strassen vs. Standard compare:          %s\n", pass_strassen_standard ? "PASS" : "FAIL");

        // --- Compute speedups ---
        double speedup_strassen = time_strassen_serial / time_strassen_parallel;
        double speedup_standard = time_standard_serial / time_standard_parallel;
        printf("Strassen speedup (Serial vs Parallel):  %.2f\n", speedup_strassen);
        printf("Standard speedup (Serial vs Parallel):  %.2f\n", speedup_standard);

        // Cleanup
        free(A);
        free(B);
        free(C_strassen_parallel);
        free(C_strassen_serial);
        free(C_standard_parallel);
        free(C_standard_serial);

        printf("\n");
    }
    return 0;
}
