#include "matrix_non_block_vectorized_serial.h"

void multiply_standard_serial_non_block_vectorized(const double *restrict A,
                                           const double *restrict B,
                                           double *restrict C,
                                           int n) {
    int i, j, k;

    for (i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }


    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    
}
