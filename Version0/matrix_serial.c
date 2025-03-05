#include "matrix.h"
#include <stdlib.h>

// Serial standard multiplication (triple nested loop)
void multiply_standard_serial(const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int n) {
        int i, j, k;
        const int blockSize = 32;  // 块尺寸，可根据硬件调节
    
        for (i = 0; i < n * n; i++) {
            C[i] = 0.0;
        }
        for (int ii = 0; ii < n; ii += blockSize) {
            int i_max = (ii + blockSize > n) ? n : (ii + blockSize);
            for (int kk = 0; kk < n; kk += blockSize) {
                int k_max = (kk + blockSize > n) ? n : (kk + blockSize);
                for (int jj = 0; jj < n; jj += blockSize) {
                    int j_max = (jj + blockSize > n) ? n : (jj + blockSize);
                    for (i = ii; i < i_max; i++) {
                        for (k = kk; k < k_max; k++) {
                            double a_ik = A[i * n + k];
                            for (j = jj; j < j_max; j++) {
                                C[i * n + j] += a_ik * B[k * n + j];
                            }
                        }
                    }
                }
            }
        }
}

