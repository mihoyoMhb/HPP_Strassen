#include "matrix.h"
#include <stdlib.h>
#include <omp.h>

// Parallel standard multiplication using OpenMP parallel for
void multiply_standard_parallel(const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int n) {
        const int blockSize = 32;  // 块尺寸，可根据硬件调节

        for (int i = 0; i < n * n; i++) {
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
       #pragma omp parallel for schedule(dynamic)
       // 外层循环进行块分解，采用 i-k-j 顺序
       for (int ii = 0; ii < n; ii += blockSize) {
           int i_max = (ii + blockSize > n) ? n : ii + blockSize;
           for (int kk = 0; kk < n; kk += blockSize) {
               int k_max = (kk + blockSize > n) ? n : kk + blockSize;
               for (int jj = 0; jj < n; jj += blockSize) {
                   int j_max = (jj + blockSize > n) ? n : jj + blockSize;
   
                   // 当前块的实际尺寸
                   // int packed_rows = k_max - kk;
                   int packed_cols = j_max - jj;
                   // 分配打包数组 B_pack（在栈上分配，若块较大可考虑动态分配）
                   double B_pack[blockSize * blockSize];
   
                   // 打包矩阵 B 的子块到 B_pack 中
                   // B_pack 按行存储，行数为 packed_rows，列数为 packed_cols
                   for (int k = kk; k < k_max; ++k) {
                       for (int j = jj; j < j_max; ++j) {
                           B_pack[(k - kk) * packed_cols + (j - jj)] = B[k * n + j];
                       }
                   }
   
                   // 利用打包后的 B_pack 进行矩阵乘法计算
                   for (int i = ii; i < i_max; ++i) {
                       for (int k = kk; k < k_max; ++k) {
                           double a_ik = A[i * n + k];
                           for (int j = jj; j < j_max; ++j) {
                               // 使用 B_pack 中的数据，计算时注意偏移
                               C[i * n + j] += a_ik * B_pack[(k - kk) * packed_cols + (j - jj)];
                           }
                       }
                   }
               }
           }
       }
}



