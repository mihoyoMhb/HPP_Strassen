#include "matrix_vectorized_serial.h"

void multiply_standard_serial_vectorized(const double *restrict A,
                                           const double *restrict B,
                                           double *restrict C,
                                           int n) {
    const int blockSize = 16;

    // 初始化 C 全部置 0
    for (int idx = 0; idx < n * n; idx++) {
        C[idx] = 0.0;
    }

    // 外层分块循环
    for (int ii = 0; ii < n; ii += blockSize) {
        int i_max = (ii + blockSize > n) ? n : (ii + blockSize);
        for (int jj = 0; jj < n; jj += blockSize) {
            int j_max = (jj + blockSize > n) ? n : (jj + blockSize);
            for (int kk = 0; kk < n; kk += blockSize) {
                int k_max = (kk + blockSize > n) ? n : (kk + blockSize);

                // 对每个分块内部再做 4×4 的子块处理
                for (int iBlock = ii; iBlock < i_max; iBlock += 4) {
                    for (int jBlock = jj; jBlock < j_max; jBlock += 4) {

                        // 处理边界情况：若不足 4 行或 4 列，则采用备用内核
                        int rowBound = ((iBlock + 4) <= i_max) ? 4 : (i_max - iBlock);
                        int colBound = ((jBlock + 4) <= j_max) ? 4 : (j_max - jBlock);

                        if (rowBound < 4 || colBound < 4) {
                            double cSub[16] = {0.0};
                            for (int kBlock = kk; kBlock < k_max; kBlock++) {
                                for (int r = 0; r < rowBound; r++) {
                                    double aVal = A[(iBlock + r) * n + kBlock];
                                    for (int c = 0; c < colBound; c++) {
                                        cSub[r * 4 + c] += aVal * B[kBlock * n + (jBlock + c)];
                                    }
                                }
                            }
                            for (int r = 0; r < rowBound; r++) {
                                for (int c = 0; c < colBound; c++) {
                                    C[(iBlock + r) * n + (jBlock + c)] += cSub[r * 4 + c];
                                }
                            }
                        }
                        else {
                            // 使用 16 个寄存器变量进行 4×4 块计算
                            register double c00 = 0, c01 = 0, c02 = 0, c03 = 0;
                            register double c10 = 0, c11 = 0, c12 = 0, c13 = 0;
                            register double c20 = 0, c21 = 0, c22 = 0, c23 = 0;
                            register double c30 = 0, c31 = 0, c32 = 0, c33 = 0;

                            for (int kBlock = kk; kBlock < k_max; kBlock++) {
                                // 加载 A 中 4 行的数据
                                register double a0 = A[(iBlock + 0) * n + kBlock];
                                register double a1 = A[(iBlock + 1) * n + kBlock];
                                register double a2 = A[(iBlock + 2) * n + kBlock];
                                register double a3 = A[(iBlock + 3) * n + kBlock];

                                // 加载 B 中对应 4 列的数据
                                register double b0 = B[kBlock * n + (jBlock + 0)];
                                register double b1 = B[kBlock * n + (jBlock + 1)];
                                register double b2 = B[kBlock * n + (jBlock + 2)];
                                register double b3 = B[kBlock * n + (jBlock + 3)];

                                c00 += a0 * b0;
                                c01 += a0 * b1;
                                c02 += a0 * b2;
                                c03 += a0 * b3;

                                c10 += a1 * b0;
                                c11 += a1 * b1;
                                c12 += a1 * b2;
                                c13 += a1 * b3;

                                c20 += a2 * b0;
                                c21 += a2 * b1;
                                c22 += a2 * b2;
                                c23 += a2 * b3;

                                c30 += a3 * b0;
                                c31 += a3 * b1;
                                c32 += a3 * b2;
                                c33 += a3 * b3;
                            } // end for kBlock

                            // 将累积结果写回 C
                            C[(iBlock + 0) * n + (jBlock + 0)] += c00;
                            C[(iBlock + 0) * n + (jBlock + 1)] += c01;
                            C[(iBlock + 0) * n + (jBlock + 2)] += c02;
                            C[(iBlock + 0) * n + (jBlock + 3)] += c03;

                            C[(iBlock + 1) * n + (jBlock + 0)] += c10;
                            C[(iBlock + 1) * n + (jBlock + 1)] += c11;
                            C[(iBlock + 1) * n + (jBlock + 2)] += c12;
                            C[(iBlock + 1) * n + (jBlock + 3)] += c13;

                            C[(iBlock + 2) * n + (jBlock + 0)] += c20;
                            C[(iBlock + 2) * n + (jBlock + 1)] += c21;
                            C[(iBlock + 2) * n + (jBlock + 2)] += c22;
                            C[(iBlock + 2) * n + (jBlock + 3)] += c23;

                            C[(iBlock + 3) * n + (jBlock + 0)] += c30;
                            C[(iBlock + 3) * n + (jBlock + 1)] += c31;
                            C[(iBlock + 3) * n + (jBlock + 2)] += c32;
                            C[(iBlock + 3) * n + (jBlock + 3)] += c33;
                        } // end if
                    } // end for jBlock
                } // end for iBlock
            }
        }
    }
}
