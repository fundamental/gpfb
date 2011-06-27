#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include "siggen.h"
#include <err.h>

void show_mem_header(void)
{
    puts("free\ttotal\tused");
}
void show_mem(void)
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
#define CUFFT_LIMIT (1<<25)
    if(transform_size*batches > CUFFT_LIMIT)
        fprintf(stderr, "Warning: CUFFT_LIMIT exceeded, please reduce "
                "batches\n");
    // Setup
    cufftHandle plan;
    checked(cufftPlan1d(&plan, transform_size, CUFFT_R2C, batches));

    // Perform FFT
    checked(cufftExecR2C(plan, src, (cufftComplex *)dest));

    // Cleanup
    checked(cufftDestroy(plan));

    return dest;
}

//Location is base address + offset
#define LOC const size_t i =\
                            (gridDim.y*(blockIdx.y*gridDim.x+blockIdx.x))+threadIdx.x

__global__ void cu_quantize(uint8_t *dest, const float *src, size_t N, size_t
        chans)
{
    LOC;
    if(i<N)
        dest[i] = quantize(src[i])*16/chans/4;
}

__global__ void cu_unquantize(float *dest, const uint8_t *src, size_t N)
{
    LOC;
    if(i<N)
        dest[i] = unquantize(src[i]);
}

__global__ void cu_movement(float *dest, const float *src, size_t N, size_t chans)
{
    LOC;
    if(i<N)
        dest[i+2*(i/chans)] = src[i];
}

__global__ void convolve(float *coeff, size_t N, float *src, size_t M, float *dest, size_t chans)
{
    LOC;
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
void apply_fir(uint8_t *buf, size_t N, const float *fir, size_t M, size_t chans,
        float *hack)
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
    checked(cudaMalloc((void **)&cu_smps, (N+2*N/chans)*sizeof(float)));
    checked(cudaMalloc((void **)&cu_buf, (N+2*N/chans+M)*sizeof(float))); 

    //Send
    //puts("Sending...");
    checked(cudaMemcpy(cu_fir, fir, M*sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(cu_bitty, buf, N, cudaMemcpyHostToDevice));
    //Buffer with zeros
    checked(cudaMemset(cu_buf, 0, M*sizeof(float)));

/* FIXME these could be changed with device/cuda version
 *       get dynamically from deviceinfo API
 *       blocks: 1024x1024x64
 *       grids:  65535x65535x65535
 */
#define MAX_BLOCK 1<<10
//using powers of 2
#define MAX_GRID 1<<15
//macro assumes variables/literals as input
#define div_up(top,bot) ((top)/(bot)+((top)%(bot)==0 ? 0:1))

    //Run
    //puts("Filtering...");
    //Calculate dimensions
    const size_t block_x = MAX_BLOCK,
                 grid_y = N/MAX_BLOCK > MAX_BLOCK ? MAX_BLOCK : 1,
                 grid_x  = div_up(N,block_x*grid_y);
    const dim3   block(block_x, 1, 1),
                 grid(grid_x, grid_y, 1);
    assert(grid.z==1);

    printf("block(%d, %d, %d) grid(%d, %d, %d) for %ld elms.\n", 
            block.x, block.y, block.z, 
            grid.x,  grid.y,  grid.z, N);

    //Convert to floating point
    cu_unquantize<<<grid, block>>>(cu_buf+M, cu_bitty, N);

    convolve<<<grid, block>>>(cu_fir, M, cu_buf+M, N, cu_smps, chans);

    cu_movement<<<grid, block>>>(cu_buf, cu_smps, N, chans);
    checked(cudaDeviceSynchronize());

    //Post Process
    //puts("FFT...");
    apply_fft(cu_buf, cu_buf, chans, N/chans);
    checked(cudaDeviceSynchronize());

    //cu_sanity<<<grid, block>>>(cu_buf,N);
    checked(cudaDeviceSynchronize());
    if(hack) checked(cudaMemcpy(hack, cu_buf, N*sizeof(float), cudaMemcpyDeviceToHost));
    //Convert to fixed point
    cu_quantize<<<grid, block>>>(cu_bitty, cu_buf, N, chans);
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
    apply_fir(buf, N, coeff, taps, chans,buffer);
    //apply_unquantize(buffer, buf, N);
    delete[] buf;

    //rescale w/ fudge factor
    //for(size_t i=0; i<N; ++i)
    //    buffer[i] *= chans;
}
