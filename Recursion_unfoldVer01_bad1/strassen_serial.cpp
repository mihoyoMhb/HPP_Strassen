#include "strassen_serial.h"
#include <cstring>
#include <memory>
#include <algorithm>

namespace {
// Helper functions in anonymous namespace (private to this file)
void add_matrix_stride(const double* A, int strideA,
                      const double* B, int strideB,
                      double* C, int strideC, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*strideC + j] = A[i*strideA + j] + B[i*strideB + j];
        }
    }
}

void sub_matrix_stride(const double* A, int strideA,
                      const double* B, int strideB,
                      double* C, int strideC, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*strideC + j] = A[i*strideA + j] - B[i*strideB + j];
        }
    }
}

// Helper function for Strassen algorithm implementation
void strassen_helper_func(double* A, int strideA,
                         double* B, int strideB,
                         double* C, int strideC,
                         int n, double** mem_ptr) {
    if (n <= BASE_SIZE) {
        multiply_standard_stride(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    const int new_n = n/2;
    const size_t sub_size = new_n * new_n;

    // Store the current pointer for the next usage
    double* const curr_ptr = *mem_ptr;

    // Allocate temporary matrices
    double *T1 = *mem_ptr; *mem_ptr += sub_size;
    double *T2 = *mem_ptr; *mem_ptr += sub_size;
    double *T3 = *mem_ptr; *mem_ptr += sub_size;
    double *T4 = *mem_ptr; *mem_ptr += sub_size;
    double *T5 = *mem_ptr; *mem_ptr += sub_size;
    double *T6 = *mem_ptr; *mem_ptr += sub_size;
    double *T7 = *mem_ptr; *mem_ptr += sub_size;
    double *M1 = *mem_ptr; *mem_ptr += sub_size;
    double *M2 = *mem_ptr; *mem_ptr += sub_size;
    double *M3 = *mem_ptr; *mem_ptr += sub_size;
    double *M4 = *mem_ptr; *mem_ptr += sub_size;
    double *M5 = *mem_ptr; *mem_ptr += sub_size;
    double *M6 = *mem_ptr; *mem_ptr += sub_size;
    double *M7 = *mem_ptr; *mem_ptr += sub_size;

    // Define submatrix views (no data copying)
    double *A11 = A;
    double *A12 = A + new_n;
    double *A21 = A + new_n * strideA;
    double *A22 = A21 + new_n;
    
    double *B11 = B;
    double *B12 = B + new_n;
    double *B21 = B + new_n * strideB;
    double *B22 = B21 + new_n;

    // M1 = (A11 + A22)(B11 + B22)
    add_matrix_stride(A11, strideA, A22, strideA, T1, new_n, new_n);
    add_matrix_stride(B11, strideB, B22, strideB, T2, new_n, new_n);
    strassen_helper_func(T1, new_n, T2, new_n, M1, new_n, new_n, mem_ptr);

    // M2 = (A21 + A22)B11
    add_matrix_stride(A21, strideA, A22, strideA, T3, new_n, new_n);
    strassen_helper_func(T3, new_n, B11, strideB, M2, new_n, new_n, mem_ptr);

    // M3 = A11(B12 - B22)
    sub_matrix_stride(B12, strideB, B22, strideB, T4, new_n, new_n);
    strassen_helper_func(A11, strideA, T4, new_n, M3, new_n, new_n, mem_ptr);

    // M4 = A22(B21 - B11)
    sub_matrix_stride(B21, strideB, B11, strideB, T5, new_n, new_n);
    strassen_helper_func(A22, strideA, T5, new_n, M4, new_n, new_n, mem_ptr);

    // M5 = (A11 + A12)B22
    add_matrix_stride(A11, strideA, A12, strideA, T6, new_n, new_n);
    strassen_helper_func(T6, new_n, B22, strideB, M5, new_n, new_n, mem_ptr);

    // M6 = (A21 - A11)(B11 + B12)
    sub_matrix_stride(A21, strideA, A11, strideA, T7, new_n, new_n);
    add_matrix_stride(B11, strideB, B12, strideB, T1, new_n, new_n);
    strassen_helper_func(T7, new_n, T1, new_n, M6, new_n, new_n, mem_ptr);

    // M7 = (A12 - A22)(B21 + B22)
    sub_matrix_stride(A12, strideA, A22, strideA, T2, new_n, new_n);
    add_matrix_stride(B21, strideB, B22, strideB, T3, new_n, new_n);
    strassen_helper_func(T2, new_n, T3, new_n, M7, new_n, new_n, mem_ptr);

    // C11 = M1 + M4 - M5 + M7
    add_matrix_stride(M1, new_n, M4, new_n, T1, new_n, new_n);
    sub_matrix_stride(T1, new_n, M5, new_n, T2, new_n, new_n);
    add_matrix_stride(T2, new_n, M7, new_n, T1, new_n, new_n);
    
    // C12 = M3 + M5
    add_matrix_stride(M3, new_n, M5, new_n, T3, new_n, new_n);
    
    // C21 = M2 + M4 
    add_matrix_stride(M2, new_n, M4, new_n, T4, new_n, new_n);
    
    // C22 = M1 - M2 + M3 + M6
    sub_matrix_stride(M1, new_n, M2, new_n, T5, new_n, new_n);
    add_matrix_stride(T5, new_n, M3, new_n, T6, new_n, new_n);
    add_matrix_stride(T6, new_n, M6, new_n, T7, new_n, new_n);

    // Combine results into the C matrix
    for (int i = 0; i < new_n; i++) {
        std::memcpy(&C[i*strideC], &T1[i*new_n], new_n*sizeof(double));
        std::memcpy(&C[i*strideC + new_n], &T3[i*new_n], new_n*sizeof(double));
        std::memcpy(&C[(i+new_n)*strideC], &T4[i*new_n], new_n*sizeof(double));
        std::memcpy(&C[(i+new_n)*strideC + new_n], &T7[i*new_n], new_n*sizeof(double));
    }

    *mem_ptr = curr_ptr;
}

} // end anonymous namespace

// Public function implementation
void multiply_standard_stride(const double* A, int strideA,
                            const double* B, int strideB,
                            double* C, int strideC, int n) {
    // Initialize C to zero
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * strideC + j] = 0;
        }
    }
    
    // Standard matrix multiplication with strides
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i * strideA + k];
            for (int j = 0; j < n; j++) {
                C[i * strideC + j] += a * B[k * strideB + j];
            }
        }
    }
}

// // 改进后的带步长矩阵乘法：分块 + 打包（B的子块）技术
// void multiply_standard_stride(const double* A, int strideA,
//                               const double* B, int strideB,
//                               double* C, int strideC, int n) {
//     const int blockSize = 64; // 根据缓存特性选择合适的块大小

//     // 初始化 C 数组为0
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             C[i * strideC + j] = 0.0;
//         }
//     }
    
//     // 分块矩阵乘法，采用 i-k-j 顺序
//     for (int ii = 0; ii < n; ii += blockSize) {
//         int i_max = (ii + blockSize > n) ? n : ii + blockSize;
//         for (int kk = 0; kk < n; kk += blockSize) {
//             int k_max = (kk + blockSize > n) ? n : kk + blockSize;
//             for (int jj = 0; jj < n; jj += blockSize) {
//                 int j_max = (jj + blockSize > n) ? n : jj + blockSize;
                
//                 // 当前块在 B 中的列数
//                 int packed_cols = j_max - jj;
//                 // 在栈上分配打包数组 B_pack，若块较大时可改为动态分配
//                 double B_pack[blockSize * blockSize];
                
//                 // 将矩阵 B 对应的子块打包到 B_pack 中
//                 // 注意：B 使用步长 strideB
//                 for (int k = kk; k < k_max; ++k) {
//                     for (int j = jj; j < j_max; ++j) {
//                         B_pack[(k - kk) * packed_cols + (j - jj)] = B[k * strideB + j];
//                     }
//                 }
                
//                 // 利用打包后的 B_pack 进行子块矩阵乘法计算
//                 // A 使用步长 strideA, C 使用步长 strideC
//                 for (int i = ii; i < i_max; ++i) {
//                     for (int k = kk; k < k_max; ++k) {
//                         double a = A[i * strideA + k];
//                         for (int j = jj; j < j_max; ++j) {
//                             C[i * strideC + j] += a * B_pack[(k - kk) * packed_cols + (j - jj)];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }


void strassen_serial_optimized(double *A, double *B, double *C, int n) {
    // Use std::unique_ptr with custom deleter for automatic memory cleanup
    size_t total_mem = 28 * n * n * sizeof(double);
    
    // Use aligned_alloc for memory alignment (important for SIMD performance)
    std::unique_ptr<double, decltype(&std::free)> mem_pool_ptr(
        static_cast<double*>(std::aligned_alloc(64, total_mem)),
        std::free
    );
    
    if (!mem_pool_ptr) {
        throw std::bad_alloc();
    }
    
    // Zero the result matrix
    std::memset(C, 0, n * n * sizeof(double));
    
    // Start the recursive computation
    double* mem_ptr = mem_pool_ptr.get();
    strassen_helper_func(A, n, B, n, C, n, n, &mem_ptr);
    
    // Memory automatically freed when mem_pool_ptr goes out of scope
}