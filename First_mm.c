#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BASE_SIZE 64  // 基准情况：小于等于此尺寸时采用标准乘法

// 函数声明
void multiply_standard(double *A, double *B, double *C, int n);
void add_matrix(double *A, double *B, double *C, int n);
void sub_matrix(double *A, double *B, double *C, int n);
void strassen(double *A, double *B, double *C, int n);

int main() {
    int n = 512;  // 假定 n 为 2 的幂
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));
    
    // 初始化矩阵 A、B（此处用随机数）
    for (int i = 0; i < n * n; i++){
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }
    
    // 利用 OpenMP 并行区域启动 Strassen 算法
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassen(A, B, C, n);
        }
    }
    
    // 这里可以添加对结果矩阵 C 的验证或输出

    free(A);
    free(B);
    free(C);
    return 0;
}

// 标准矩阵乘法：三重循环实现
void multiply_standard(double *A, double *B, double *C, int n) {
    int i, j, k;
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            C[i * n + j] = 0;
            for(k = 0; k < n; k++){
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 矩阵加法：C = A + B
void add_matrix(double *A, double *B, double *C, int n) {
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

// 矩阵减法：C = A - B
void sub_matrix(double *A, double *B, double *C, int n) {
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

// Strassen 算法的并行实现，矩阵均以行优先一维数组存储，尺寸 n x n
void strassen(double *A, double *B, double *C, int n) {
    if(n <= BASE_SIZE) {
        multiply_standard(A, B, C, n);
        return;
    }
    
    int newSize = n / 2;
    int size = newSize * newSize;
    int i, j;
    
    // 分配 A 的四个子矩阵
    double *A11 = (double*)malloc(size * sizeof(double));
    double *A12 = (double*)malloc(size * sizeof(double));
    double *A21 = (double*)malloc(size * sizeof(double));
    double *A22 = (double*)malloc(size * sizeof(double));
    
    // 分配 B 的四个子矩阵
    double *B11 = (double*)malloc(size * sizeof(double));
    double *B12 = (double*)malloc(size * sizeof(double));
    double *B21 = (double*)malloc(size * sizeof(double));
    double *B22 = (double*)malloc(size * sizeof(double));
    
    // 将 A 和 B 拆分为子矩阵
    for(i = 0; i < newSize; i++){
        for(j = 0; j < newSize; j++){
            A11[i * newSize + j] = A[i * n + j];
            A12[i * newSize + j] = A[i * n + j + newSize];
            A21[i * newSize + j] = A[(i + newSize) * n + j];
            A22[i * newSize + j] = A[(i + newSize) * n + j + newSize];
            
            B11[i * newSize + j] = B[i * n + j];
            B12[i * newSize + j] = B[i * n + j + newSize];
            B21[i * newSize + j] = B[(i + newSize) * n + j];
            B22[i * newSize + j] = B[(i + newSize) * n + j + newSize];
        }
    }
    
    // 为 M1 ... M7 以及临时矩阵分配内存
    double *M1 = (double*)malloc(size * sizeof(double));
    double *M2 = (double*)malloc(size * sizeof(double));
    double *M3 = (double*)malloc(size * sizeof(double));
    double *M4 = (double*)malloc(size * sizeof(double));
    double *M5 = (double*)malloc(size * sizeof(double));
    double *M6 = (double*)malloc(size * sizeof(double));
    double *M7 = (double*)malloc(size * sizeof(double));
    
    double *T1 = (double*)malloc(size * sizeof(double));
    double *T2 = (double*)malloc(size * sizeof(double));
    double *T3 = (double*)malloc(size * sizeof(double));
    double *T4 = (double*)malloc(size * sizeof(double));
    double *T5 = (double*)malloc(size * sizeof(double));
    double *T6 = (double*)malloc(size * sizeof(double));
    double *T7 = (double*)malloc(size * sizeof(double));
    double *T8 = (double*)malloc(size * sizeof(double));
    double *T9 = (double*)malloc(size * sizeof(double));
    double *T10 = (double*)malloc(size * sizeof(double));
    
    // 按照 Strassen 公式构造各个子问题：
    // M1 = (A11 + A22) * (B11 + B22)
    add_matrix(A11, A22, T1, newSize); // T1 = A11 + A22
    add_matrix(B11, B22, T2, newSize); // T2 = B11 + B22
    
    // M2 = (A21 + A22) * B11
    add_matrix(A21, A22, T3, newSize); // T3 = A21 + A22
    
    // M3 = A11 * (B12 - B22)
    sub_matrix(B12, B22, T4, newSize); // T4 = B12 - B22
    
    // M4 = A22 * (B21 - B11)
    sub_matrix(B21, B11, T5, newSize); // T5 = B21 - B11
    
    // M5 = (A11 + A12) * B22
    add_matrix(A11, A12, T6, newSize); // T6 = A11 + A12
    
    // M6 = (A21 - A11) * (B11 + B12)
    sub_matrix(A21, A11, T7, newSize); // T7 = A21 - A11
    add_matrix(B11, B12, T8, newSize); // T8 = B11 + B12
    
    // M7 = (A12 - A22) * (B21 + B22)
    sub_matrix(A12, A22, T9, newSize); // T9 = A12 - A22
    add_matrix(B21, B22, T10, newSize); // T10 = B21 + B22
    
    // 利用 OpenMP task 并行计算 7 个子乘法
    #pragma omp task shared(M1) firstprivate(T1, T2, newSize)
    {
        strassen(T1, T2, M1, newSize);
    }
    #pragma omp task shared(M2) firstprivate(T3, B11, newSize)
    {
        strassen(T3, B11, M2, newSize);
    }
    #pragma omp task shared(M3) firstprivate(A11, T4, newSize)
    {
        strassen(A11, T4, M3, newSize);
    }
    #pragma omp task shared(M4) firstprivate(A22, T5, newSize)
    {
        strassen(A22, T5, M4, newSize);
    }
    #pragma omp task shared(M5) firstprivate(T6, B22, newSize)
    {
        strassen(T6, B22, M5, newSize);
    }
    #pragma omp task shared(M6) firstprivate(T7, T8, newSize)
    {
        strassen(T7, T8, M6, newSize);
    }
    #pragma omp task shared(M7) firstprivate(T9, T10, newSize)
    {
        strassen(T9, T10, M7, newSize);
    }
    #pragma omp taskwait  // 等待所有任务完成
    
    // 组合子问题结果得到 C 的四个象限
    double *C11 = (double*)malloc(size * sizeof(double));
    double *C12 = (double*)malloc(size * sizeof(double));
    double *C21 = (double*)malloc(size * sizeof(double));
    double *C22 = (double*)malloc(size * sizeof(double));
    
    // C11 = M1 + M4 - M5 + M7
    add_matrix(M1, M4, C11, newSize);
    sub_matrix(C11, M5, C11, newSize);
    add_matrix(C11, M7, C11, newSize);
    
    // C12 = M3 + M5
    add_matrix(M3, M5, C12, newSize);
    
    // C21 = M2 + M4
    add_matrix(M2, M4, C21, newSize);
    
    // C22 = M1 - M2 + M3 + M6
    sub_matrix(M1, M2, C22, newSize);
    add_matrix(C22, M3, C22, newSize);
    add_matrix(C22, M6, C22, newSize);
    
    // 将 C11, C12, C21, C22 合并到结果矩阵 C 中
    for(i = 0; i < newSize; i++){
        for(j = 0; j < newSize; j++){
            C[i * n + j] = C11[i * newSize + j];
            C[i * n + j + newSize] = C12[i * newSize + j];
            C[(i + newSize) * n + j] = C21[i * newSize + j];
            C[(i + newSize) * n + j + newSize] = C22[i * newSize + j];
        }
    }
    
    // 释放所有临时分配的内存
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2); free(T3); free(T4); free(T5); free(T6);
    free(T7); free(T8); free(T9); free(T10);
    free(C11); free(C12); free(C21); free(C22);
}
