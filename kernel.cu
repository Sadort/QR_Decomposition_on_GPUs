//#include <stdio.h>
//#include <stdlib.h>
//#include <cuda_runtime.h>
//#include "cublas_v2.h"
#include "kernel.h"
#include <assert.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void update_diagonal(double *d_v, double *d_x, double *d_norm)
{ 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 1)
    {
        if (*d_x >= 0) {
            d_v[0] = *d_x - *d_norm;
        }else{
            d_v[0] = *d_x + *d_norm;
        }
    }
}

__global__ void update_beta(double *d_beta, double *d_dotpro)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 1)
    {
        *d_beta = (double)(-2 / *d_dotpro);
    }
}

__global__ void mycopy(double *W, double *V, double *beta, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < len)
    {
        W[tid] = (*beta) * V[tid];
    }

}

__global__ void initial_float(double *in, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < len)
    {
        in[tid] = 0.0;
    }

//    if(tid * 4 + 4 <= len)
//    {
//        in[tid * 4] = 0.0f;
//        in[tid * 4 + 1] = 0.0f;
//        in[tid * 4 + 2] = 0.0f;
//        in[tid * 4 + 3] = 0.0f;
//        in[tid * 8 + 4] = 0.0f;
//        in[tid * 8 + 5] = 0.0f;
//        in[tid * 8 + 6] = 0.0f;
//        in[tid * 8 + 7] = 0.0f;
//    }
//    if(len % 4 == tid)
//    {
//        in[len - (len % 4) + tid] = 0.0f;
//    }
}

/*
Computes W = beta A V + alpha W
*/
//__global__ void gtSgemv(float alpha, float beta, float *A, float *V, int M, int N, float *W)


void house(cublasHandle_t handle, double *d_v, double *d_x, double *d_beta, int len, int m, int n)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;

    double *d_norm, *d_dotpro;
    double ZERO = 0.0;
    
    cudaStat = cudaMalloc((void**)&d_norm, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&d_dotpro, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);

    cudaStat = cudaMemcpy(d_norm, &ZERO, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(d_dotpro, &ZERO, sizeof(double) * 1, cudaMemcpyHostToDevice);  
    assert(cudaSuccess == cudaStat);

    cudaDeviceSynchronize();

    stat = cublasDnrm2(handle, len, d_x, 1, d_norm);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    stat = cublasDcopy(handle, len, d_x, 1, d_v, 1);
    //mycopy<<<1024*16, 256>>>(d_v, d_x, len);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    update_diagonal<<<1, 8>>>(d_v, d_x, d_norm);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    stat = cublasDdot(handle, len, d_v, 1, d_v, 1, d_dotpro);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    update_beta<<<1, 8>>>(d_beta, d_dotpro);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    cudaFree(d_norm);
    cudaFree(d_dotpro);
    
    return;
}

void apply_house(cublasHandle_t handle, double *d_v, double *d_A, double *d_beta, int len, int m, int n, int r)
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
    double ONE = 1.0, ZERO = 0.0;

    double *d_v_A;
    cudaStat = cudaMalloc((void**)&d_v_A, sizeof(double) * sub_n);
    //cudaStat = cudaMemset((void*)d_v_A, ZERO, sizeof(float) * sub_n);
    assert(cudaSuccess == cudaStat);
 
    initial_float<<<1024*64, 1024>>>(d_v_A, sub_n);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    double *alpha;
    double *beta;
    cudaStat = cudaMalloc((void**)&alpha, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&beta, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaDeviceSynchronize();

    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, sub_n, sub_m, alpha, d_v, 1, d_A, m, beta, d_v_A, 1);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    cudaStat = cudaMemcpy(beta, &ONE, sizeof(double) * 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat);

    stat = cublasDger(handle, sub_m, sub_n, d_beta, d_v, 1, d_v_A, 1, d_A, m);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    cudaFree(d_v_A);
    cudaFree(alpha);
    cudaFree(beta);    
    return;

}

void generate_WY(cublasHandle_t handle, double *W, const double *Y, double *d_beta, int m, int n, int len, int r)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
   
    double ONE = 1.0, ZERO = 0.0;
    double *d_Y_v;
    double *alpha;
    double *beta;
    cudaStat = cudaMalloc((void**)&d_Y_v, sizeof(double) * r);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&alpha, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&beta, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaDeviceSynchronize();

    initial_float<<<1024*64, 1024>>>(d_Y_v, r);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    //mycopy<<<1024*16, 256>>>(Y, W, d_beta, len);
    stat = cublasDaxpy(handle, len, d_beta, Y, 1, W, 1);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);
    //stat = cublasSaxpy(handle, len*(r-1), alpha, &Y[IDX2C(0,1,len)], 1, &W[IDX2C(0,1,len)], 1);

    stat = cublasDcopy(handle, len*(r-1), &Y[IDX2C(0,1,len)], 1, &W[IDX2C(0,1,len)], 1);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    for (int j = 1; j < r; j++) {
        stat = cublasDgemv(handle, CUBLAS_OP_T, len, j, alpha, Y, len, &Y[IDX2C(0,j,len)], 1, beta, d_Y_v, 1);
        cudaDeviceSynchronize();
        assert(stat == CUBLAS_STATUS_SUCCESS);

        stat = cublasDgemv(handle, CUBLAS_OP_N, len, j, &d_beta[j], W, len, d_Y_v, 1, &d_beta[j], &W[IDX2C(0,j,len)], 1);
        cudaDeviceSynchronize();
        assert(stat == CUBLAS_STATUS_SUCCESS);

    }

    cudaFree(d_Y_v);
    cudaFree(alpha);
    cudaFree(beta);

    return;
}

void apply_WY(cublasHandle_t handle, double *d_A, double *W, double *Y, int m, int n, int len, int r)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    
    double *d_W_A;
    double *alpha;
    double *beta;
    double ONE = 1.0, ZERO = 0.0;
    int sub_m = len;
    int sub_n = n - m + len;
    cudaStat = cudaMalloc((void**)&d_W_A, sizeof(double) * r * sub_n);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&alpha, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&beta, sizeof(double) * 1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(alpha, &ONE, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(beta, &ZERO, sizeof(double) * 1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);

    initial_float<<<1024*64, 1024>>>(d_W_A, r*sub_n);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    cudaDeviceSynchronize();
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r, sub_n, sub_m, alpha, W, sub_m, d_A, m, beta, d_W_A, r);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, r, alpha, Y, sub_m, d_W_A, r, alpha, d_A, m);
    cudaDeviceSynchronize();
    assert(stat == CUBLAS_STATUS_SUCCESS);

    cudaFree(d_W_A);
    cudaFree(alpha);
    cudaFree(beta);

    return;
}

void unblocked_qr_calculate(double *d_A, int m, int n)
{
    cudaError_t cudaStat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    int len = 0;
    double *d_beta, *d_house_v;
    cudaStat = cudaMalloc((void**)&d_beta,sizeof(double)*1);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&d_house_v,sizeof(double)*m);
    assert(cudaSuccess == cudaStat);
    cudaDeviceSynchronize();

    initial_float<<<1024*64, 1024>>>(d_house_v, m);
    cudaDeviceSynchronize();

    assert(cudaGetLastError() == cudaSuccess);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0, housetime = 0, applytime = 0;

    for (int k = 0; k < n; k++) {

        //householder reflector
        len = m - k;
      
        cudaEventRecord(start);

        //printf("%d access house()\n", k);
        house(handle, d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        housetime += milliseconds;

        cudaEventRecord(start);

        //apply householder reflector
        //printf("%d access apply_house()\n", k);
        apply_house(handle, d_house_v, &d_A[IDX2C(k,k,m)], d_beta, len, m, n, 1);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        applytime += milliseconds;

    }
    printf("house: %f \n", housetime);
    printf("apply house: %f \n", applytime);

    cudaFree(d_beta);
    cudaFree(d_house_v);
    return;
}

void blocked_qr_calculate(double *d_A, int m, int n, int r)
{
    if (n % r != 0) {
        return;
    }
    int num_block = n / r;
    cudaError_t cudaStat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    int len = 0, sub_len;
    int first_row_ind = 0, ind;
    double *d_beta, *d_house_v;
    double *W;
    
    cudaStat = cudaMalloc((void**)&d_beta,sizeof(double)*r);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&d_house_v,sizeof(double)*m*r);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc((void**)&W,sizeof(double)*m*r);
    assert(cudaSuccess == cudaStat);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0, housetime = 0, applytime = 0, WYtime = 0, applyWYtime = 0;

    for (int k = 0; k < num_block; k++) {
        first_row_ind = k * r;
        len = m - first_row_ind;
        initial_float<<<1024*64, 1024>>>(d_beta, r);
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);
        initial_float<<<1024*64, 1024>>>(d_house_v, len*r);
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);
        initial_float<<<1024*64, 1024>>>(W, len*r);
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);

        for (int j = 0; j < r; j++) {
            ind = first_row_ind + j;
            sub_len = len - j;

            cudaEventRecord(start);

            //printf("block %d, row %d, access house()\n", k, j);
            //householder reflector
            house(handle, &d_house_v[IDX2C(j,j,len)], &d_A[IDX2C(ind,ind,m)], &d_beta[j], sub_len, m, n);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            housetime += milliseconds;

            cudaEventRecord(start);

            //printf("block %d, row %d, access apply_house()\n", k, j);
            //apply householder reflector
            apply_house(handle, &d_house_v[IDX2C(j,j,len)], &d_A[IDX2C(ind,ind,m)], &d_beta[j], sub_len, m, n, r);
 
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            applytime += milliseconds;

        }
        if(len == m - n + r)
            break;

        cudaEventRecord(start);

        //printf("block %d, access grenerate_WY()\n", k);
        generate_WY(handle, W, d_house_v, d_beta, m, n, len, r);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        WYtime += milliseconds;

        cudaEventRecord(start);

        //apply W & Y
        //printf("block %d, access apply_WY()\n", k);
        apply_WY(handle, &d_A[IDX2C(first_row_ind,first_row_ind+r,m)], W, d_house_v, m, n, len, r);
    
        cudaEventRecord(stop);         
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        applyWYtime += milliseconds;

    }

    printf("house: %f \n", housetime);
    printf("apply house: %f \n", applytime);
    printf("WY: %f \n", WYtime);
    printf("apply WY: %f \n", applyWYtime);

    cudaFree(W);
    cudaFree(d_house_v);
    cudaFree(d_beta);
    return;
}
