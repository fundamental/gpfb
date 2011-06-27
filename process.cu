#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include <stdint.h>
#include "siggen.h"
#include <err.h>

void show_mem_header(void)
{
    puts("free\ttotal\tused");
}
void show_mem(const char *func, int line)
{
    //int nbytes=100000;
    float free_m,total_m,used_m;
    size_t free,total;

    cudaMemGetInfo((size_t*)&free,(size_t*)&total);

    free_m =free/1048576.0 ;
    total_m=total/1048576.0;
    used_m=(total_m-free_m);
    printf ( "%f\t%f\t%f\n", free_m,total_m,used_m);
}

#define checked(x) { cufftResult_t e = x; if(e) {\
    fprintf(stderr,"CUFFT error[%s][%d]: %s\n", #x, e,\
            cudaGetErrorString(cudaGetLastError()));\
    exit(0);}}
//apply fft using out of place transform
float *apply_fft(float *src, float *dest, size_t transform_size, size_t batches)
{
    //printf("Samples to fft: %ld\n", batches*transform_size);
    assert(CUFFT_SUCCESS==0);
    // Setup
    cufftHandle plan;
    checked(cufftPlan1d(&plan, transform_size, CUFFT_R2C, batches));

    // Perform FFT
    checked(cufftExecR2C(plan, src, (cufftComplex *)dest));

    // Cleanup
    checked(cufftDestroy(plan));

    return dest;
}

__global__ void cu_quantize(uint8_t *dest, const float *src, size_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)
        dest[i] = quantize(src[i]);
}

__global__ void cu_unquantize(float *dest, const uint8_t *src, size_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)
        dest[i] = unquantize(src[i]);
}

__global__ void cu_movement(float *dest, const float *src, size_t N, size_t chans)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)
        dest[i+2*(i/chans)] = src[i];
}

__global__ void convolve(float *coeff, size_t N, float *src, size_t M, float *dest, size_t chans)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=M)
        return;

    unsigned sel = i%chans;

    //do actual work at i
    dest[i] = 0.0;
    for(size_t j=sel; j<N; j+=chans)
        dest[i] += src[i-sel-j]*coeff[j];
    dest[i] /= chans;
}

#undef checked
#define checked(x) { cudaError_t e = x; if(e!= cudaSuccess) {\
    fprintf(stderr,"CUDA error[%s]: %s\n", #x, cudaGetErrorString(e));\
    exit(-1);}}
void apply_fir(uint8_t *buf, size_t N, const float *fir, size_t M, size_t chans)
{
    //insure clean decimation
    assert(M%chans==0);
    assert(N%chans==0);

    uint8_t *cu_bitty = NULL;
    float   *cu_fir   = NULL,
            *cu_buf   = NULL,
            *cu_smps  = NULL;

    //Allocate
    //puts("Allocating...");
    checked(cudaMalloc((void **)&cu_bitty, N));
    checked(cudaMalloc((void **)&cu_fir, M*sizeof(float)));
    checked(cudaMalloc((void **)&cu_smps, N*sizeof(float)));
    checked(cudaMalloc((void **)&cu_buf, (N+M)*sizeof(cufftComplex))); 
    //TODO check to see if the complex is valid       ^^^^^^^^^^^^

    //Send
    //puts("Sending...");
    checked(cudaMemcpy(cu_fir, fir, M*sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(cu_bitty, buf, N, cudaMemcpyHostToDevice));
    //Buffer with zeros
    checked(cudaMemset(cu_buf, 0, M*sizeof(float)));

    //Run
    //puts("Filtering...");
    const int block_size = 1024,
              blocks = N/block_size + (N%block_size == 0 ? 0:1);
    //Convert to floating point
    cu_unquantize <<< blocks, block_size >>>(cu_buf+M, cu_bitty, N);
    checked(cudaDeviceSynchronize());

    convolve <<< blocks, block_size >>>(cu_fir, M, cu_buf+M, N, cu_smps, chans);
    checked(cudaDeviceSynchronize());

    cu_movement <<< blocks, block_size >>>(cu_buf, cu_smps, N, chans);
    checked(cudaDeviceSynchronize());

    //Post Process
    //puts("FFT...");
    apply_fft(cu_buf, cu_buf, chans, N/chans);
    checked(cudaDeviceSynchronize());

    //Convert to fixed point
    cu_quantize <<< blocks, block_size >>>(cu_bitty, cu_buf, N);
    checked(cudaDeviceSynchronize());

    //TODO copy back all information or discard unwanted portions, as they are
    //not needed

    //Retreive
    //puts("Getting...");
    checked(cudaMemcpy(buf, cu_bitty, N, cudaMemcpyDeviceToHost));

    //Clean
    //puts("Cleaning...");
    checked(cudaFree(cu_bitty));
    checked(cudaFree(cu_fir));
    checked(cudaFree(cu_smps));
    checked(cudaFree(cu_buf));
}

void apply_pfb(float *buffer, size_t N, float *coeff, size_t taps, size_t chans)
{
    uint8_t *buf = new uint8_t[N];
    apply_quantize(buf, buffer, N);
    apply_fir(buf, N, coeff, taps, chans);
    apply_unquantize(buffer, buf, N);
    delete[] buf;

    //rescale w/ fudge factor
    for(size_t i=0; i<N; ++i)
        buffer[i] *= chans;
}
