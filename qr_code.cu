#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void update_diagonal(float *d_v, float *d_x, float *d_norm)
{
    if (*d_x > 0) {
        *d_v = *d_x - *d_norm;
    }else{
        *d_v = *d_x + *d_norm;
    }
}

__global__ void update_beta(float *d_beta, float *d_dotpro)
{
    *d_beta = -2 / *d_dotpro;
}

/*
    Computes W = beta A V + alpha W
*/
//__global__ void gtSgemv(float alpha, float *d_beta, )

void house(float *d_v, float *d_x, float *d_beta, int len, int m, int n)
{
    printf("1\n");
    cudaError_t cudaStat;
    printf("2\n");
    cublasStatus_t stat;
    printf("3\n");
    cublasHandle_t handle;
    printf("4\n");
    cublasCreate(&handle);
    printf("5\n");
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    printf("6\n");
    float *d_norm, *d_dotpro;
    cudaStat = cudaMalloc((void**)&d_norm, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&d_dotpro, sizeof(float) * 1);
    printf("point0\n");
    cudaStat = cudaMemset(d_norm, 0, sizeof(float) * 1);
    cudaStat = cudaMemset(d_dotpro, 0, sizeof(float) * 1);
    cudaDeviceSynchronize();
    printf("point1\n");
//    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);  
    stat = cublasSnrm2(handle, len, d_x, 1, d_norm);
    cudaDeviceSynchronize();

    printf("point2\n");
//    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    stat = cublasScopy(handle, len, d_x, 1, d_v, 1);
    cudaDeviceSynchronize();
    update_diagonal<<<1, 1>>>(d_v, d_x, d_norm);
    cudaDeviceSynchronize();
    printf("point3\n");
//    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    stat = cublasSdot(handle, len, d_v, 1, d_v, 1, d_dotpro);
    cudaDeviceSynchronize();
    printf("point4\n");
    update_beta<<<1, 1>>>(d_beta, d_dotpro);
    cudaDeviceSynchronize();
    printf("point5\n");
    cublasDestroy(handle);
    cudaFree(d_norm);
    cudaFree(d_dotpro);
    
    return;
}

void apply_house(float *d_v, float *d_A, float *d_beta, int len, int m, int n)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    int sub_m = len;
    int sub_n = n - m + len;
    
    float *d_v_A;
    printf("point1\n");
    cudaStat = cudaMalloc((void**)&d_v_A, sizeof(float) * sub_n);
    printf("point2\n");
    cudaStat = cudaMemset(d_v_A, 0, sizeof(float) * sub_n);

    float *alpha;
    float *beta;
    cudaStat = cudaMalloc((void**)&alpha, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&beta, sizeof(float) * 1);
    cudaStat = cudaMemset(alpha, 1, sizeof(float) * 1);
    cudaStat = cudaMemset(beta, 0, sizeof(float) * 1);
    printf("point3\n");
    cudaDeviceSynchronize();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, sub_n, sub_m, alpha, d_v, sub_m, d_A, m, beta, d_v_A, 1);
    cudaDeviceSynchronize();

    cudaStat = cudaMemset(beta, 1, sizeof(float) * 1);
    printf("point4\n");
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, 1, d_beta, d_v, sub_m, d_v_A, 1, beta, d_A, sub_m);
    cudaDeviceSynchronize();
    printf("point5\n");

    cublasDestroy(handle);
    cudaFree(d_v_A);
    
    return;
    
}

void qr_calculate(float *d_A, int m, int n)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    //cublasHandle_t handle;
    
    int len = 0;
    float *d_beta, *d_house_v;
    cudaStat = cudaMalloc((void**)&d_beta,sizeof(float)*1);
    cudaStat = cudaMalloc((void**)&d_house_v,sizeof(float)*m);
    cudaDeviceSynchronize();

    for (int k = 0; k < n; k++) {
        
        //householder reflector
        len = m - k;
        printf("%d access house()\n", k);
        house(d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n);
        
        //apply householder reflector
        printf("%d access apply_house()\n", k);
        apply_house(d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n);
    }
}

int main()
{
    int m = 4, n = 3, i;
    float A[m*n] = { 1, 1, 1, 1, -1, 4, 4, -1, 4, -2, 2, 0 };
    float *d_A;
    
    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaStat = cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    qr_calculate(d_A, m, n);
    
    cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    
    for (i = 0; i < m*n; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");
    
    return 0;
}
