#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "matrix_vectorized_serial.h"

int main() {
    int n = 1024;  // 测试矩阵尺寸，可根据需要调整
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));

    // 使用随机数初始化 A、B（值在 [0,1]）
    srand(42);
    for (int i = 0; i < n * n; i++){
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    double start = omp_get_wtime();
    multiply_standard_serial_vectorized(A, B, C, n);
    double end = omp_get_wtime();

    printf("Vectorized serial matrix multiplication (n=%d) took %.6f seconds.\n", n, end - start);

    free(A);
    free(B);
    free(C);
    return 0;
}
