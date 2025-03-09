#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cblas.h>
#include "matrix.h"

#ifdef __AVX2__
#include <immintrin.h>
#define BLOCK_SIZE 32

// 使用 AVX2 和 FMA 指令实现的块式矩阵乘法（双精度）
void matrix_multiply_avx(double* A, double* B, double* C, int N) {
    // 先将 C 初始化为 0
    memset(C, 0, N * N * sizeof(double));
    
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            double buffer[BLOCK_SIZE][BLOCK_SIZE] = {0};
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                int i_end = (i + BLOCK_SIZE) < N ? (i + BLOCK_SIZE) : N;
                int j_end = (j + BLOCK_SIZE) < N ? (j + BLOCK_SIZE) : N;
                int k_end = (k + BLOCK_SIZE) < N ? (k + BLOCK_SIZE) : N;
                for (int kk = k; kk < k_end; ++kk) {
                    for (int ii = i; ii < i_end; ++ii) {
                        __m256d a = _mm256_set1_pd(A[ii * N + kk]);
                        int jj = j;
                        for (; jj <= j_end - 4; jj += 4) {
                            __m256d b = _mm256_loadu_pd(&B[kk * N + jj]);
                            __m256d c = _mm256_loadu_pd(&buffer[ii - i][jj - j]);
                            c = _mm256_fmadd_pd(a, b, c);
                            _mm256_storeu_pd(&buffer[ii - i][jj - j], c);
                        }
                        for (; jj < j_end; ++jj) {
                            buffer[ii - i][jj - j] += A[ii * N + kk] * B[kk * N + jj];
                        }
                    }
                }
            }
            int i_end = (i + BLOCK_SIZE) < N ? (i + BLOCK_SIZE) : N;
            int j_end = (j + BLOCK_SIZE) < N ? (j + BLOCK_SIZE) : N;
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    C[ii * N + jj] += buffer[ii - i][jj - j];
                }
            }
        }
    }
}
#else
// 如果当前平台不支持 AVX2，则回退到普通三重循环
void matrix_multiply_avx(const double* restrict A, const double* restrict B, double* C, int n) {
    printf("AVX2 not supported on this architecture. Using fallback multiplication.\n");
    memset(C, 0, n * n * sizeof(double));
    const int blockSize = 32; // 调整为适合缓存的块大小
    // 外层循环进行块分解，采用 i-k-j 顺序
    for (int ii = 0; ii < n; ii += blockSize) {
        int i_max = (ii + blockSize > n) ? n : ii + blockSize;
        for (int kk = 0; kk < n; kk += blockSize) {
            int k_max = (kk + blockSize > n) ? n : kk + blockSize;
            for (int jj = 0; jj < n; jj += blockSize) {
                int j_max = (jj + blockSize > n) ? n : jj + blockSize;

                // 当前块的实际尺寸
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
#endif

int main() {
    // 定义测试矩阵尺寸
    int test_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // 设置 OpenMP 线程数
    omp_set_num_threads(8);
    int num_threads = 8;

    printf("Comparing Parallel Strassen vs. Serial Strassen,\n");
    printf("Parallel Standard vs. Serial Standard,\n");
    printf("OpenBLAS GEMM and AVX Blocked multiplication.\n");
    printf("Number of threads: %d\n\n", num_threads);

    for (int idx = 0; idx < num_sizes; idx++) {
        int n = test_sizes[idx];
        printf("===== Matrix Size: %dx%d =====\n", n, n);

        // 分配矩阵和各算法的结果数组（全部使用 double 类型）
        double *A = (double*)malloc(n * n * sizeof(double));
        double *B = (double*)malloc(n * n * sizeof(double));
        double *C_strassen_parallel = (double*)malloc(n * n * sizeof(double));
        double *C_strassen_serial   = (double*)malloc(n * n * sizeof(double));
        double *C_standard_parallel = (double*)malloc(n * n * sizeof(double));
        double *C_standard_serial   = (double*)malloc(n * n * sizeof(double));
        double *C_gemm = (double*)malloc(n * n * sizeof(double));  // OpenBLAS GEMM 结果
        double *C_avx = (double*)malloc(n * n * sizeof(double));   // AVX Blocked 结果

        // 初始化 A 和 B（随机值在 [0,1]）
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

        // --- 2) Serial Strassen ---
        start_time = omp_get_wtime();
        strassen_serial_optimized(A, B, C_strassen_serial, n);
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

        // --- 5) OpenBLAS GEMM Multiplication ---
        start_time = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0, A, n, B, n, 0.0, C_gemm, n);
        end_time = omp_get_wtime();
        double time_gemm = end_time - start_time;
        printf("OpenBLAS GEMM time:     %.4f seconds\n", time_gemm);

        // --- 6) AVX Blocked Multiplication ---
        start_time = omp_get_wtime();
        matrix_multiply_avx(A, B, C_avx, n);
        end_time = omp_get_wtime();
        double time_avx = end_time - start_time;
        printf("AVX Blocked multiplication time: %.4f seconds\n", time_avx);

        // --- Compare correctness ---
        int pass_strassen_standard = compare_matrices(C_strassen_parallel, C_standard_serial, n);
        printf("Strassen vs. Standard compare:                %s\n", pass_strassen_standard ? "PASS" : "FAIL");
        int pass_strassen_parallel = compare_matrices(C_strassen_parallel, C_standard_parallel, n);
        printf("Parallel Strassen vs. Parallel Standard compare: %s\n", pass_strassen_parallel ? "PASS" : "FAIL");
        int pass_strassen_serial = compare_matrices(C_strassen_serial, C_standard_serial, n);
        printf("Serial Strassen vs. Serial Standard compare:     %s\n", pass_strassen_serial ? "PASS" : "FAIL");
        int pass_gemm = compare_matrices(C_gemm, C_standard_serial, n);
        printf("OpenBLAS GEMM vs. Serial Standard compare:       %s\n", pass_gemm ? "PASS" : "FAIL");
        int pass_avx = compare_matrices(C_avx, C_standard_serial, n);
        printf("AVX Blocked vs. Serial Standard compare:         %s\n", pass_avx ? "PASS" : "FAIL");

        // --- Compute speedups ---
        printf("---Speedups:---\n");
        double speedup_strassen_standard = time_standard_serial / time_strassen_serial;
        printf("Strassen vs Standard speedup (Serial):        %.2f\n", speedup_strassen_standard);
        double parallel_speedup = time_standard_serial / time_standard_parallel;
        printf("Parallel speedup (Standard serial vs Standard parallel): %.2f\n", parallel_speedup);
        double final_speedup = time_standard_serial / time_strassen_parallel;
        printf("Final speedup (Standard serial vs Strassen_parallel):  %.2f\n", final_speedup);
        double gemm_speedup = time_standard_serial / time_gemm;
        printf("GEMM speedup (Standard serial vs OpenBLAS GEMM): %.2f\n", gemm_speedup);
        double avx_speedup = time_standard_serial / time_avx;
        printf("AVX speedup (Standard serial vs AVX Blocked):     %.2f\n", avx_speedup);

        // Cleanup
        free(A);
        free(B);
        free(C_strassen_parallel);
        free(C_strassen_serial);
        free(C_standard_parallel);
        free(C_standard_serial);
        free(C_gemm);
        free(C_avx);

        printf("\n");
    }
    return 0;
}
