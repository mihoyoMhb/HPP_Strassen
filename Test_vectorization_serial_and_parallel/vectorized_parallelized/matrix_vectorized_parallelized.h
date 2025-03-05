#ifndef MATRIX_VECTORIZED_PARALLELIZED_H
#define MATRIX_VECTORIZED_PARALLELIZED_H

void multiply_standard_parallelized_vectorized(const double *restrict A,
                                           const double *restrict B,
                                           double *restrict C,
                                           int n);

#endif // MATRIX_VECTORIZED_PARALLELIZED_H