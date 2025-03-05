#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "matrix_vectorized_parallelized.h"

int main(){
    int n = 2048;
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));

    // 使用随机数初始化 A、B（值在 [0,1]）
    srand(42);
    for (int i = 0; i < n * n; i++){
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
    omp_set_num_threads(8);
    double start = omp_get_wtime();
    multiply_standard_parallelized_vectorized(A, B, C, n);
    double end = omp_get_wtime();
    printf("Vectorized parallelized matrix multiplication (n=%d) took %.6f seconds.\n", n, end - start);
    free(A);
    free(B);
    free(C);
    return 0;
}