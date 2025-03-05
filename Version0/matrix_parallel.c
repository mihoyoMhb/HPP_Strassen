#include "matrix.h"
#include <stdlib.h>
#include <omp.h>

// Parallel standard multiplication using OpenMP parallel for
void multiply_standard_parallel(const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int n) {
        int i, j, k;
        const int blockSize = 16;  // 块尺寸，可根据硬件调节
    
        for (i = 0; i < n * n; i++) {
            C[i] = 0.0;
        }
    
        /*
        Block Matrix Multiplication
    
        The core idea of block matrix multiplication is partitioned computation, 
        where each computation processes a small submatrix at a time. 
        This ensures that frequently accessed data stays in the CPU cache as long as 
        possible, reducing cache misses.
    
        Optimized Approach:
    
        1. Block Partitioning: 
        
        The matrix is divided into smaller blockSize × blockSize 
        submatrices, and each block is processed separately.
        In-block Computation: Each iteration only processes a smaller submatrix, 
        which improves cache hit rates.
    
        2. Optimized Memory Access Order: 
    
        The original code uses row-major storage, 
        and the access pattern follows row-wise traversal, 
        making it more compatible with CPU cache prefetching mechanisms.
    
        https://www.reddit.com/r/learnprogramming/comments/le4ve/how_does_blocking_increase_speedup_of_matrix/
        https://ximera.osu.edu/la/LinearAlgebra/MAT-M-0023/main
        https://www.netlib.org/utk/papers/autoblock/node2.html
        */
        #pragma omp parallel for schedule(static)
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



