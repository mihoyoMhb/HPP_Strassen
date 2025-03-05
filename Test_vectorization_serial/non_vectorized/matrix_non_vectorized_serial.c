#include "matrix_non_vectorized_serial.h"

void multiply_standard_serial_non_vectorized(const double *A,
                                               const double *B,
                                               double *C,
                                               int n) {
    int i, j, k;
    const int blockSize = 64;  

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
