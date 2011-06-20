#include <cuda.h>
#include <cufft.h>
#include <assert.h>
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

__global__ void convolve(float *coeff, size_t N, float *src, size_t M, float *dest, size_t chans)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>M)
        return;

    unsigned sel = i%chans;

    //do actual work at i
    dest[i] = 0.0;
    for(size_t j=sel; j<N; j+=chans)
        dest[i] += src[i-sel-j]*coeff[j];
}

#undef checked
#define checked(x) { if(x!=cudaSuccess) err(1, #x);}
float *apply_fir(float *buf, size_t N, float *coeff, size_t M, size_t chans)
{
    //insure clean decimation
    assert(M%chans==0);
    assert(N%chans==0);

    float *cu_coeff=NULL, *cu_buf=NULL, *cu_smps=NULL;//r,r,w

    //Allocate
    //puts("Allocating...");
    checked(cudaMalloc((void **)&cu_coeff, M*sizeof(float)));
    checked(cudaMalloc((void **)&cu_smps, N*sizeof(float)));
    checked(cudaMalloc((void **)&cu_buf, (N+M)*sizeof(cufftComplex)));

    //Send
    //puts("Sending...");
    checked(cudaMemcpy(cu_coeff, coeff, M*sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(cu_buf, buf-M, (N+M)*sizeof(float), cudaMemcpyHostToDevice));

    //Run
    //puts("Filtering...");
    int block_size = 128;
    int blocks = N/block_size + (N%block_size == 0 ? 0:1);
    convolve <<< blocks, block_size >>>(cu_coeff, M, cu_buf+M, N, cu_smps, chans);
    cudaDeviceSynchronize();

    //Post Process
    //puts("FFT...");
    apply_fft(cu_smps, cu_buf, chans, N/chans);
    cudaDeviceSynchronize();

    //Retreive
    //puts("Getting...");
    checked(cudaMemcpy(buf, cu_buf, sizeof(float)*N, cudaMemcpyDeviceToHost));

    //Clean
    //puts("Cleaning...");
    checked(cudaFree(cu_coeff));
    checked(cudaFree(cu_smps));
    checked(cudaFree(cu_buf));
    return buf;

}

void apply_pfb(float *buffer, size_t N, float *coeff, size_t taps, size_t chans)
{
    apply_fir(buffer, N, coeff, taps, chans);
}
