#include "matrix.h"

/*
Strassen Parallel Code Structure (C++ Version)
├── main() / strassenLoopParallel(A, B, C, n)
│   ├── 【内存池分配】 
│   │   └── 使用 aligned_alloc 分配 mem_pool（满足 SIMD 对齐要求）
│   ├── 【任务栈初始化】
│   │   └── 初始化线程安全的 std::stack<TaskFrame>
│   ├── 【OpenMP 并行区域】
│   │   └── #pragma omp parallel
│   │         └── #pragma omp single
│   │               └── 循环：while(!taskStack.empty())
│   │                     └── #pragma omp task untied
│   │                           ├── 判断任务：if(frame.n <= BASE_SIZE)
│   │                           │      └── 调用 multiplyStandardStride()（基准串行乘法）
│   │                           └── else
│   │                                  └── 调用 processStrassenLayer(frame, mem_pool)
│   └── 【释放内存池】（利用 RAII 或在并行区域结束后调用 free）
│
├── Data Structure: TaskFrame
│   ├── double* A, B, C         → 矩阵数据指针
│   ├── int strideA, strideB, strideC  → 矩阵步长（分块存储用）
│   ├── int n                   → 当前子矩阵尺寸
│   └── int stage               → 任务阶段（0：分解阶段；1：组合阶段；…）
│
├── processStrassenLayer(TaskFrame* frame, double* mem_pool)
│   ├── 计算 new_n = frame->n / 2；sub_size = new_n * new_n
│   ├── 【分层内存分配】
│   │      └── 调用 get_layer_offset(frame->n, initial_n)
│   │            返回偏移量，用于定位当前层的临时空间 temp_blocks
│   ├── 【定义中间矩阵指针】
│   │      └── 如 M1 = temp_blocks, M2 = M1 + sub_size, …（根据算法需要）
│   └── 【状态机判断】
│          ├── case 0（初始分解阶段）：调用 createSubtasks(frame, temp_blocks)
│          └── case 1（组合阶段）：调用 combineResults(frame, temp_blocks)
│
├── get_layer_offset(int current_n, int initial_n)
│   └── 功能：
│         └── 从初始尺寸 initial_n 递归降至 current_n，每层累计所需内存空间
│               └── 具体逻辑：
│                     while (n > current_n) {
│                         offset += 14 * (n/2) * (n/2);
│                         n /= 2;
│                     }
│                     返回 offset（单位为 double 数量）
│
├── createSubtasks(TaskFrame* frame, double* temp)
│   ├── 为 7 个中间矩阵（M1～M7）计算创建独立的 OpenMP 任务
│   │      └── 每个任务内部调用 compute_M_matrix(i, frame, temp)（或相似函数）
│   └── 创建组合任务：
│          └── 利用 #pragma omp task depend 声明依赖关系，
│                待 7 个任务全部完成后更新 frame->stage 为 1，并重新将任务帧入栈
│
├── combineResults(TaskFrame* frame, double* temp)
│   └── 功能：
│         └── 将 7 个中间矩阵组合成子矩阵，写回到结果矩阵 C 中
│
├── multiplyStandardStride(const double* A, int strideA,
│       const double* B, int strideB,
│       double* C, int strideC, int n)
│   ├── 【初始化】
│   │      └── 将 C 的所有元素设为 0
│   └── 【三重循环计算】
│          └── i、k、j 三层循环，完成标准矩阵乘法计算
│
├── blockMultiply(MatrixBlock a, MatrixBlock b, MatrixBlock c)
│   └── 功能：
│         └── 利用块状存储优化内存访问，采用 #pragma omp parallel for collapse(2)
│               实现高缓存局部性的块矩阵乘法
│
└── multiplyBase(const double* A, const double* B, double* C, int n)
    └── 功能：
          └── 利用 SIMD 向量化（AVX-512）加速基础矩阵乘法，
                使用 #pragma omp parallel for simd collapse(2)，
                注意数据对齐和边界处理

*/


void strassen_loop_parallel(double* A, double* B, double* C, int n){
    
    // The temporary space for the Strassen algorithm
    // The size of the temporary space is 28 * n * n doubles
    size_t total_mem = 28 * n * n * sizeof(double);
    double *mem_pool = (double*)aligned_alloc(64, total_mem);
    
    #pragma omp parallel
    #pragma omp single
    {
        
        TaskFrame root = {A, B, C, n, n, 0};
        // Push the root task frame onto the stack
        // The stack opeations shoudle be protected by a critical section
        #pragma omp critical
        {
            TaskStack.push(root);
        }

        // The main loop of the Strassen algorithm
        while(!TaskStack.empty()){
            #pragma omp task untied
            {
                TaskFrame current_frame;
                #pragma omp critical
                {
                    current_frame = TaskStack.top();
                    TaskStack.pop();
                }
                if(current_frame.n <= BASE_SIZE) {
                    multiply_standard_stride(current_frame.A, current_frame.strideA,
                                             current_frame.B, current_frame.strideB,
                                             current_frame.C, current_frame.strideC,
                                             current_frame.n);
                }else{
                    process_strassen_layer(&current_frame, mem_pool);
                }
            }
        }

    }
    // Free the memory pool at the end
    free(mem_pool);
}


void processStrassenLayer(TaskFrame* frame, double* mem_pool){
    /*Pre-allocate memory for each recursive level to avoid dynamic allocation overhead. 
    A state machine is also used here to distribute tasks.*/
}


size_t get_layer_offset(int current_n, int initial_n) {
    size_t offset = 0;
    int n = initial_n;
    
    // 遍历从初始尺寸到当前尺寸的所有层级
    while (n > current_n) {
        // 每层需要 14 个临时矩阵，每个尺寸为 (n/2)^2
        offset += 14 * (n / 2) * (n / 2);
        n /= 2;
    }
    return offset;
}
