#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include <stdint.h>
#include "siggen.h"
#include <err.h>

#define checked(x) { cufftResult_t e = x; if(e) err(e, #x);}
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
#define checked(x) { if(x!=cudaSuccess) err(1, #x);}
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
    const int block_size = 128,
              blocks = N/block_size + (N%block_size == 0 ? 0:1);
    //Convert to floating point
    cu_unquantize <<< blocks, block_size >>>(cu_buf+M, cu_bitty, N);
    cudaDeviceSynchronize();

    convolve <<< blocks, block_size >>>(cu_fir, M, cu_buf+M, N, cu_smps, chans);
    cudaDeviceSynchronize();

    //Post Process
    //puts("FFT...");
    apply_fft(cu_smps, cu_buf, chans, N/chans);
    cudaDeviceSynchronize();

    //Convert to fixed point
    cu_quantize <<< blocks, block_size >>>(cu_bitty, cu_buf, N);
    cudaDeviceSynchronize();

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
    uint8_t buf[N];
    apply_quantize(buf, buffer, N);
    apply_fir(buf, N, coeff, taps, chans);
    apply_unquantize(buffer, buf, N);

    //rescale w/ fudge factor
    for(size_t i=0; i<N; ++i)
        buffer[i] *= chans;
}
