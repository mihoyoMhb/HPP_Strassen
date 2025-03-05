#ifndef MATRIX_VECTORIZED_SERIAL_H
#define MATRIX_VECTORIZED_SERIAL_H

// Optimized (vectorized) serial matrix multiplication for square matrices.
// Computes C = A * B. Matrices are stored in row-major order.
// This version uses the restrict qualifier and a SIMD directive to help vectorization.
void multiply_standard_serial_vectorized(const double *restrict A,
                                           const double *restrict B,
                                           double *restrict C,
                                           int n);

#endif // MATRIX_VECTORIZED_SERIAL_H
