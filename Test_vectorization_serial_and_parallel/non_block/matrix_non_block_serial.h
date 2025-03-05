#ifndef MATRIX_NON_VECTORIZED_SERIAL_H
#define MATRIX_NON_VECTORIZED_SERIAL_H

// Non-vectorized serial matrix multiplication for square matrices.
// Computes C = A * B, with matrices stored in row-major order.
// This version uses a simple blocking technique without restrict 或 SIMD 指令.
void multiply_standard_serial_non_block(const double *A,
                                               const double *B,
                                               double *C,
                                               int n);

#endif // MATRIX_NON_VECTORIZED_SERIAL_H
