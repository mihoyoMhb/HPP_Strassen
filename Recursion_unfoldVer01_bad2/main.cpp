#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include "matrix.h"
#include "strassen_serial.h"

// Function to generate random matrices
void generateRandomMatrix(double* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dis(gen);
    }
}

// Function to compute standard matrix multiplication for reference
void standard_matrix_multiply(const double* A,
    const double*  B,
    double *C,
    int n) {
        const int blockSize = 64;
        // // 初始化 C 数组
        // for (int i = 0; i < n * n; ++i) {
        //     C[i] = 0.0;
        // }
    
        // // 外层循环进行块分解，采用 i-k-j 顺序
        // for (int ii = 0; ii < n; ii += blockSize) {
        //     int i_max = (ii + blockSize > n) ? n : ii + blockSize;
        //     for (int kk = 0; kk < n; kk += blockSize) {
        //         int k_max = (kk + blockSize > n) ? n : kk + blockSize;
        //         for (int jj = 0; jj < n; jj += blockSize) {
        //             int j_max = (jj + blockSize > n) ? n : jj + blockSize;
    
        //             // 当前块的实际尺寸
        //             int packed_cols = j_max - jj;
        //             // 分配打包数组 B_pack（在栈上分配，若块较大可考虑动态分配）
        //             double B_pack[blockSize * blockSize];
    
        //             // 打包矩阵 B 的子块到 B_pack 中
        //             // B_pack 按行存储，行数为 packed_rows，列数为 packed_cols
        //             for (int k = kk; k < k_max; ++k) {
        //                 for (int j = jj; j < j_max; ++j) {
        //                     B_pack[(k - kk) * packed_cols + (j - jj)] = B[k * n + j];
        //                 }
        //             }
    
        //             // 利用打包后的 B_pack 进行矩阵乘法计算
        //             for (int i = ii; i < i_max; ++i) {
        //                 for (int k = kk; k < k_max; ++k) {
        //                     double a_ik = A[i * n + k];
        //                     for (int j = jj; j < j_max; ++j) {
        //                         // 使用 B_pack 中的数据，计算时注意偏移
        //                         C[i * n + j] += a_ik * B_pack[(k - kk) * packed_cols + (j - jj)];
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
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

// Function to calculate the Frobenius norm of the difference between two matrices
double matrix_difference_norm(double* A, double* B, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++) {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to print a small portion of a matrix for visual check
void print_matrix_sample(double* matrix, int n, int sample_size = 3) {
    int size = std::min(sample_size, n);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "(Showing " << size << "x" << size << " sample of " << n << "x" << n << " matrix)" << std::endl;
}


void debug_print_matrix_info(double* mat, int n, const char* name) {
    std::cout << "Matrix " << name << " info:" << std::endl;
    std::cout << "  Address: " << mat << std::endl;
    std::cout << "  Size: " << n << "x" << n << std::endl;
    std::cout << "  First few elements: ";
    for (int i = 0; i < std::min(5, n*n); i++) {
        std::cout << mat[i] << " ";
    }
    std::cout << std::endl;
}

// Add this at the beginning of your main() function before calling strassen_loop_parallel
void debug_matrices(double* A, double* B, double* C, int n) {
    debug_print_matrix_info(A, n, "A");
    debug_print_matrix_info(B, n, "B");
    debug_print_matrix_info(C, n, "C");
    
    // Test basic operations
    std::cout << "Testing basic matrix operations..." << std::endl;
    
    // Can we write to C?
    try {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] = 0.0;
            }
        }
        std::cout << "Successfully zeroed matrix C" << std::endl;
    }
    catch (...) {
        std::cerr << "Failed to write to matrix C" << std::endl;
    }
    
    // Test reading from A and B
    try {
        double sum_a = 0.0, sum_b = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum_a += A[i * n + j];
                sum_b += B[i * n + j];
            }
        }
        std::cout << "Sum of A: " << sum_a << ", Sum of B: " << sum_b << std::endl;
    }
    catch (...) {
        std::cerr << "Failed to read from matrices A or B" << std::endl;
    }
}

// Call this function in main before running strassen_loop_parallel



int main() {
    std::cout << "Strassen Matrix Multiplication Comparison" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Test with various matrix sizes
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}; // Add more sizes as needed
    
    for (int n : sizes) {
        std::cout << "\nTesting with matrix size n = " << n << std::endl;
        
        // Allocate memory for matrices
        double *A = (double*)aligned_alloc(64, n * n * sizeof(double));
        double *B = (double*)aligned_alloc(64, n * n * sizeof(double));
        double *C_serial = (double*)aligned_alloc(64, n * n * sizeof(double));
        double *C_parallel = (double*)aligned_alloc(64, n * n * sizeof(double));
        double *C_standard = (double*)aligned_alloc(64, n * n * sizeof(double));
        
        if (!A || !B || !C_serial || !C_parallel || !C_standard) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1;
        }
        std::cout << "Memory allocated successfully." << std::endl;
        // Generate random input matrices
        generateRandomMatrix(A, n);
        generateRandomMatrix(B, n);
        std::cout << "Random matrices generated." << std::endl;
        // Compute standard matrix multiplication for reference
        auto start_std = std::chrono::high_resolution_clock::now();
        standard_matrix_multiply(A, B, C_standard, n);
        auto end_std = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_std = end_std - start_std;
        
        // Run the serial Strassen algorithm
        auto start_serial = std::chrono::high_resolution_clock::now();
        strassen_serial_optimized(A, B, C_serial, n);
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_serial = end_serial - start_serial;
        
        std::cout << "Serial Strassen multiplication completed." << std::endl;

        // Run the parallel Strassen algorithm
        auto start_parallel = std::chrono::high_resolution_clock::now();
        //debug_matrices(A, B, C_parallel, n);
        strassen_parallel_optimized(A, B, C_parallel, n);
        auto end_parallel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_parallel = end_parallel - start_parallel;
        
        // Calculate error
        double error_serial = matrix_difference_norm(C_serial, C_standard, n);
        double error_parallel = matrix_difference_norm(C_parallel, C_standard, n);
        
        // Print results
        std::cout << "Standard multiplication time: " << elapsed_std.count() << " seconds" << std::endl;
        std::cout << "Serial Strassen time: " << elapsed_serial.count() << " seconds (speedup: " 
                  << elapsed_std.count() / elapsed_serial.count() << "x)" << std::endl;
        std::cout << "Parallel Strassen time: " << elapsed_parallel.count() << " seconds (speedup: " 
                   << elapsed_std.count() / elapsed_parallel.count() << "x)" << std::endl;
        // std::cout << "Serial vs Parallel speedup: " << elapsed_serial.count() / elapsed_parallel.count() << "x" << std::endl;
        
        std::cout << "Error of serial Strassen: " << error_serial << std::endl;
         std::cout << "Error of parallel Strassen: " << error_parallel << std::endl;
        
        // // Print a small sample of the matrices for visual check
        // if (n <= 512) { // Only for smaller matrices
        //     std::cout << "\nSample of the result matrices:" << std::endl;
            
        //     std::cout << "\nStandard multiplication result:" << std::endl;
        //     print_matrix_sample(C_standard, n);
            
        //     std::cout << "\nSerial Strassen result:" << std::endl;
        //     print_matrix_sample(C_serial, n);
            
        //     std::cout << "\nParallel Strassen result:" << std::endl;
        //     print_matrix_sample(C_parallel, n);
        // }
        
        // Free memory
        free(A);
        free(B);
        free(C_serial);
        free(C_parallel);
        free(C_standard);
    }
    
    std::cout << "\nComparison complete!" << std::endl;
    return 0;
}