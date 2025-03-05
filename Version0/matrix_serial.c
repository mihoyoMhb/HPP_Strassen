#include "matrix.h"
#include <stdlib.h>

// Serial standard multiplication (triple nested loop)
void multiply_standard_serial(const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int n) {
        const int blockSize = 32; // 调整为适合缓存的块大小

        // 初始化 C 数组
        for (int i = 0; i < n * n; ++i) {
            C[i] = 0.0;
        }
    
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

