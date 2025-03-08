#ifndef STRASSEN_SERIAL_H
#define STRASSEN_SERIAL_H

// Base size for which we switch to standard multiplication
#define BASE_SIZE 512

/**
 * Standard matrix multiplication with stride support
 * 
 * @param A Source matrix A
 * @param strideA Row stride of matrix A
 * @param B Source matrix B
 * @param strideB Row stride of matrix B
 * @param C Destination matrix C = A*B
 * @param strideC Row stride of matrix C
 * @param n Matrix dimension (n x n)
 */
void multiply_standard_stride(const double* A, int strideA,
                            const double* B, int strideB,
                            double* C, int strideC, int n);

/**
 * Optimized serial implementation of Strassen's matrix multiplication algorithm
 * 
 * @param A Source matrix A (n x n)
 * @param B Source matrix B (n x n)
 * @param C Destination matrix C = A*B (n x n)
 * @param n Matrix dimension
 */
void strassen_serial_optimized(double* A, double* B, double* C, int n);

#endif // STRASSEN_SERIAL_H