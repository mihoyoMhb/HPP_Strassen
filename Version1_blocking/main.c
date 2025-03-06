#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matrix.h"

int main() {
    // Define the matrix sizes for testing
    int test_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
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

        int pass_strassen_standard = compare_matrices(C_strassen_parallel, C_standard_serial, n);
        printf("Strassen vs. Standard compare:          %s\n", pass_strassen_standard ? "PASS" : "FAIL");
        int pass_strassen_parallel = compare_matrices(C_strassen_parallel, C_standard_parallel, n);
        printf("Parallel Strassen vs. Parallel Standard compare:   %s\n", pass_strassen_parallel ? "PASS" : "FAIL");
        int pass_strassen_serial = compare_matrices(C_strassen_serial, C_standard_serial, n);
        printf("Serial Strassen vs. Serial Standard compare:       %s\n", pass_strassen_serial ? "PASS" : "FAIL");

        
        // --- Compute speedups ---
        printf("---Speedups:---\n");
        double speedup_strassen_standard = time_standard_serial/time_strassen_serial;
        printf("Strassen vs Standard speedup (Serial):  %.2f\n", speedup_strassen_standard);
        double prarallel_speedup = time_standard_serial/time_standard_parallel;
        printf("Parallel speedup (Standard serial vs Standard parallel): %.2f\n", prarallel_speedup);
        double final_speedup = time_standard_serial/time_strassen_parallel;
        printf("Final speedup (Standard vs Strassen_parallel): %.2f\n", final_speedup);

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
