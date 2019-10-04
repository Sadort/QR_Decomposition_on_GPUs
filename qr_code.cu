#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void update_diagonal(float *d_v, float *d_x, float *d_norm)
{
    if(threadIdx.x == 1)
    {
        if (*d_x >= 0) {
            d_v[0] = *d_x - *d_norm;
        }else{
            d_v[0] = *d_x + *d_norm;
        }
    }
}

__global__ void update_beta(float *d_beta, float *d_dotpro)
{
    if(threadIdx.x == 1)
    {
        *d_beta = (float)(-2 / *d_dotpro);
    }
}

__global__ void mycopy(float *W, float *V, float *beta, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < len)
    {
        W[tid] = (*beta) * V[tid];
    }

}

/*
Computes W = beta A V + alpha W
*/
//__global__ void gtSgemv(float alpha, float beta, float *A, float *V, int M, int N, float *W)


void house(cublasHandle_t handle, float *d_v, float *d_x, float *d_beta, int len, int m, int n)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;

    float *d_norm, *d_dotpro;
    float ZERO = 0;
    
    cudaStat = cudaMalloc((void**)&d_norm, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&d_dotpro, sizeof(float) * 1);
  
    cudaStat = cudaMemcpy(d_norm, &ZERO, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(d_dotpro, &ZERO, sizeof(float) * 1, cudaMemcpyHostToDevice);  

    cudaDeviceSynchronize();

    stat = cublasSnrm2(handle, len, d_x, 1, d_norm);
    cudaDeviceSynchronize();

    stat = cublasScopy(handle, len, d_x, 1, d_v, 1);
    //mycopy<<<1024*16, 256>>>(d_v, d_x, len);

    cudaDeviceSynchronize();
    update_diagonal<<<1, 8>>>(d_v, d_x, d_norm);
    cudaDeviceSynchronize();
    
    stat = cublasSdot(handle, len, d_v, 1, d_v, 1, d_dotpro);
    cudaDeviceSynchronize();
    
    update_beta<<<1, 8>>>(d_beta, d_dotpro);
    cudaDeviceSynchronize();
    

    cudaFree(d_norm);
    cudaFree(d_dotpro);
    
    return;
}

void apply_house(cublasHandle_t handle, float *d_v, float *d_A, float *d_beta, int len, int m, int n, int r)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    
    int sub_m = len;
    int sub_n;
    if (r == 1) {
        sub_n = n - m + len;
    }else{
        sub_n = (len % r == 0 ? r : (len % r));
    }
    float ONE = 1, ZERO = 0;

    float *d_v_A;
    cudaStat = cudaMalloc((void**)&d_v_A, sizeof(float) * sub_n);
    //cudaStat = cudaMemset((void*)d_v_A, ZERO, sizeof(float) * sub_n);

    float *alpha;
    float *beta;
    cudaStat = cudaMalloc((void**)&alpha, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&beta, sizeof(float) * 1);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    
    cudaDeviceSynchronize();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, sub_n, sub_m, alpha, d_v, 1, d_A, m, beta, d_v_A, 1);
    cudaDeviceSynchronize();
    

    cudaStat = cudaMemcpy(beta, &ONE, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    
    stat = cublasSger(handle, sub_m, sub_n, d_beta, d_v, 1, d_v_A, 1, d_A, m);
    cudaDeviceSynchronize();
    
    cudaFree(d_v_A);
    
    return;

}

void generate_WY(cublasHandle_t handle, float *W, float *Y, float *d_beta, int m, int n, int len, int r)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
   
    int ONE = 1, ZERO = 0;
    float *d_Y_v;
    float *alpha;
    float *beta;
    cudaStat = cudaMalloc((void**)&d_Y_v, sizeof(float) * r);
    cudaStat = cudaMalloc((void**)&alpha, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&beta, sizeof(float) * 1);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(float) * 1, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    
    //mycopy<<<1024*16, 256>>>(Y, W, d_beta, len);
    stat = cublasSaxpy(handle, len, d_beta, Y, 1, W, 1);
    cudaDeviceSynchronize();
    
    for (int j = 1; j < r; j++) {
        stat = cublasSgemv(handle, CUBLAS_OP_T, len, j, alpha, Y, len, &Y[IDX2C(0,j,len)], len, beta, d_Y_v, j);
        cudaDeviceSynchronize();
        
        stat = cublasSgemv(handle, CUBLAS_OP_N, len, j, &d_beta[j], W, len, d_Y_v, j, &d_beta[j], &W[IDX2C(0,j,len)], len);
        cudaDeviceSynchronize();
    }

    cudaFree(d_Y_v);
    cudaFree(alpha);
    cudaFree(beta);

    return;
}

void apply_WY(cublasHandle_t handle, float *d_A, float *W, float *Y, int m, int n, int len, int r)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    
    float *d_W_A;
    float *alpha;
    float *beta;
    int ONE = 1, ZERO = 0;
    int sub_m = len;
    int sub_n = n - m + len;
    cudaStat = cudaMalloc((void**)&d_W_A, sizeof(float) * r * sub_n);
    cudaStat = cudaMalloc((void**)&alpha, sizeof(float) * 1);
    cudaStat = cudaMalloc((void**)&beta, sizeof(float) * 1);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(float) * 1, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r, sub_n, sub_m, alpha, W, sub_m, d_A, m, beta, d_W_A, r);
    cudaDeviceSynchronize();

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, r, alpha, Y, sub_m, d_W_A, r, alpha, d_A, m);
    cudaDeviceSynchronize();


    cudaFree(d_W_A);
    cudaFree(alpha);
    cudaFree(beta);

    return;
}

void unblocked_qr_calculate(float *d_A, int m, int n)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    int len = 0;
    float *d_beta, *d_house_v;
    cudaStat = cudaMalloc((void**)&d_beta,sizeof(float)*1);
    cudaStat = cudaMalloc((void**)&d_house_v,sizeof(float)*m);
    cudaDeviceSynchronize();

    for (int k = 0; k < n; k++) {

        //householder reflector
        len = m - k;
        printf("%d access house()\n", k);
        house(handle, d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n);

        //apply householder reflector
        printf("%d access apply_house()\n", k);
        apply_house(handle, d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n, 1);
    }
}

void blocked_qr_calculate(float *d_A, int m, int n, int r)
{
    if (n % r != 0) {
        return;
    }
    int num_block = n / r;
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    int len = 0, ZERO = 0, sub_len;
    int first_row_ind = 0, ind;
    float *d_beta, *d_house_v;
    float *W, *Y;
    float A[m*n] = { 1, 1, 1, 1, -1, 4, 4, -1, 4, -2, 2, 0, 1, 2, 2, 3 };
    float SW[m*r] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    cudaStat = cudaMalloc((void**)&d_beta,sizeof(float)*r);
    cudaStat = cudaMalloc((void**)&d_house_v,sizeof(float)*m*r);
    cudaStat = cudaMalloc((void**)&W,sizeof(float)*m*r);
    cudaDeviceSynchronize();


    for (int k = 0; k < num_block; k++) {
        first_row_ind = k * r;
        len = m - first_row_ind;
        //printf("block %d\n", k);
        cudaStat = cudaMemset((void*)d_beta, ZERO, sizeof(float) * r);
        cudaStat = cudaMemset((void*)d_house_v, ZERO, sizeof(float) * len * r);
        cudaStat = cudaMemset((void*)W, ZERO, sizeof(float) * len * r);
        
        cudaDeviceSynchronize();

        for (int j = 0; j < r; j++) {
            ind = first_row_ind + j;
            sub_len = len - j;
            printf("block %d, row %d, access house()\n", k, j);
            //householder reflector
            house(handle, &d_house_v[IDX2C(j,j,len)], &d_A[IDX2C(ind,ind,m)], &d_beta[j], sub_len, m, n);
            printf("block %d, row %d, access apply_house()\n", k, j);
            //apply householder reflector
            apply_house(handle, &d_house_v[IDX2C(j,j,len)], &d_A[IDX2C(ind,ind,m)], &d_beta[j], sub_len, m, n, r);
            /*cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for (int i = 0; i < m*n; i++) {
                printf("%f ", A[i]);
            }
            printf("\n");*/
        }
        if(len == m - n + r)
            return;

        printf("block %d, access grenerate_WY()\n", k);
        generate_WY(handle, W, d_house_v, d_beta, m, n, len, r);
        cudaDeviceSynchronize();
        cudaStat = cudaMemcpy(SW, d_house_v, sizeof(float)*m*r, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (int i = 0; i < m*r; i++) {
            printf("%f ", SW[i]);
        }
        printf("\n");

        //apply W & Y
        printf("block %d, access apply_WY()\n", k);
        apply_WY(handle, &d_A[IDX2C(first_row_ind,first_row_ind+r,m)], W, Y, m, n, len, r);
        
    }
}

int main()
{
    int m = 4, n = 4, i;
    int r = 1;
    float A[m*n] = { 1, 1, 1, 1, -1, 4, 4, -1, 4, -2, 2, 0, 1, 2, 2, 3 };
    float *d_A;

    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaStat = cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

//    unblocked_qr_calculate(d_A, m, n);

    r = 2;
    blocked_qr_calculate(d_A, m, n, r);

    cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (i = 0; i < m*n; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");

    return 0;
}
