#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include "siggen.h"
#include "process.h"
#include <err.h>
#include <iostream>

//Function Declarations
void apply_polyphase(uint8_t *buf, Pfb &pfb, float *hack=NULL);

//Macro Definitions
#define checked(x) { cudaError_t e = x; if(e!= cudaSuccess) {\
    fprintf(stderr,"CUDA error[%s][%d]: %s\n", #x, __LINE__, cudaGetErrorString(e));\
    exit(-1);}}

#define fftchecked(x) { cufftResult_t e = x; if(e) {\
    fprintf(stderr,"CUFFT error[%s][%d]: %s\n", #x, e,\
            cudaGetErrorString(cudaGetLastError()));\
    exit(0);}}

//Class Definitions
struct Pfb
{
    void i_pfb(const float *fir, size_t _nFir, size_t _nSmps, size_t _nChan);
    void d_pfb(void);
    
    //Call when device buffer is unused
    void run(uint8_t *data) { 
        apply_polyphase(data, *this);
        checked(cudaDeviceSynchronize());
    };

    //Parameter sizes
    size_t nFir, nSmps, nChan;

    //Device Buffers
    uint8_t *bitty;
    float   *fir,
            *buf,
            *smps;

    //Host buffer
    uint8_t *h_buf;

    //FFT handle
    cufftHandle plan;
};

void Pfb::i_pfb(const float *_fir, size_t _nFir, size_t _nSmps, size_t _nChan)
{
    nFir  = _nFir;
    nSmps = _nSmps;
    nChan = _nChan;

    //insure clean decimation
    assert((nFir  % nChan) == 0);
    assert((nSmps % nChan) == 0);

    //Allocate GPU buffers
    checked(cudaMalloc((void **)&bitty, nSmps));
    checked(cudaMalloc((void **)&fir, nFir*sizeof(float)));
    size_t nStrideData = nSmps+2*nSmps/nChan;
    checked(cudaMalloc((void **)&smps,(nStrideData+nFir)*sizeof(float)));
    checked(cudaMalloc((void **)&buf, (nStrideData+nFir)*sizeof(float)));

    //Send over FIR data [TODO re-evaluate for const memory]
    checked(cudaMemcpy(fir, _fir, nFir*sizeof(float), cudaMemcpyHostToDevice));

    //Allocate CPU buffer
    checked(cudaHostAlloc((void**) &h_buf, nSmps, cudaHostAllocDefault));

    //Allocate FFT
#define CUFFT_LIMIT (1<<27)
    if(nSmps > CUFFT_LIMIT)
        fprintf(stderr, "Warning: CUFFT_LIMIT exceeded, please reduce "
                "batches\n");
    // Setup
    fftchecked(cufftPlan1d(&plan, nChan, CUFFT_R2C, nSmps/nChan));
    //fftchecked(cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
}

void Pfb::d_pfb(void)
{
    //Clean up gpu memory
    checked(cudaFree(bitty));
    checked(cudaFree(fir));
    checked(cudaFree(smps));
    checked(cudaFree(buf));

    //Clean up host memory
    checked(cudaFreeHost(h_buf));

    // Cleanup
    fftchecked(cufftDestroy(plan));
}

std::ostream &operator<<(std::ostream &out, const Pfb &p)
{
    using namespace std;
#define prnt(x) out << #x ": " << (void *) x << endl;
    prnt(p.nFir);
    prnt(p.nSmps);
    prnt(p.nChan);
    prnt(p.bitty);
    prnt(p.fir);
    prnt(p.buf);
    prnt(p.smps);
    return out;
}

class Pfb *alloc_pfb(const float *fir, size_t _nFir, size_t _nSmps, size_t _nChan)
{
    Pfb *p = new Pfb;
    p->i_pfb(fir, _nFir, _nSmps, _nChan);
    return p;
}

void delete_pfb(class Pfb *p)
{
    p->d_pfb();
    delete p;
}

//Helper Functions
void show_mem_header(void)
{
    puts("free\ttotal\tused");
}
void show_mem(void)
{
    float free_m,total_m,used_m;
    size_t free,total;

    cudaMemGetInfo((size_t*)&free,(size_t*)&total);

    free_m =free/1048576.0 ;
    total_m=total/1048576.0;
    used_m=(total_m-free_m);
    printf ( "%f\t%f\t%f\n", free_m,total_m,used_m);
}

//assumes smps -> buf
void fft_pad(Pfb &pfb)
{
    //Use Aliases for clarity
    const size_t width  = pfb.nChan,
          height = pfb.nSmps/pfb.nChan,
          sFloat = sizeof(float);

    float *dest = pfb.buf;
    const float *src = pfb.smps;

    checked(cudaMemcpy2D(dest, (width+2)*sFloat, src, width*sFloat,
                width*sFloat, height, cudaMemcpyDeviceToDevice));
}

//apply fft using out of place transform
float *apply_fft(float *src, float *dest, Pfb &pfb)
{
    fft_pad(pfb);
    // Perform FFT
    fftchecked(cufftExecR2C(pfb.plan, src, (cufftComplex *)dest));
    return dest;
}

//Main Kernel code 

//Location is base address + offset
#define LOC const size_t i =\
                            (gridDim.y*(blockIdx.y*gridDim.x+blockIdx.x))+threadIdx.x

__global__ void cu_quantize(uint8_t *dest, const float *src, size_t N, size_t
        chans)
{
    LOC;
    if(i<N)
        dest[i] = quantize(src[i]*2);
}

__global__ void cu_unquantize(float *dest, const uint8_t *src, size_t N)
{
    LOC;
    if(i<N)
        dest[i] = unquantize(src[i]);
}

__global__ void convolve(float *dest, const float *src, const float *coeff,
        size_t nC, size_t nS, size_t chan)
{
    LOC;
    if (i<nS) {
        unsigned     sel   = i%chan;

        //do actual work at i
        float result = 0.0f;
#pragma unroll
        for(size_t j=sel; j<nC; j+=chan)
            result += src[i-sel-j]*coeff[j];
        dest[i] = result/chan;
    }
}

void apply_polyphase(uint8_t *buf, Pfb &pfb, float *hack)
{
    checked(cudaMemcpyAsync(pfb.bitty, buf, pfb.nSmps, cudaMemcpyHostToDevice));
    //Buffer with zeros
    checked(cudaMemset(pfb.buf, 0, pfb.nFir*sizeof(float)));

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
                 grid_y = pfb.nSmps/MAX_BLOCK > MAX_BLOCK ? MAX_BLOCK : 1,
                 grid_x  = div_up(pfb.nSmps,block_x*grid_y);
    const dim3   block(block_x, 1, 1),
                 grid(grid_x, grid_y, 1);

    if(0)
        printf("thread(%d, %d, %d) block(%d, %d, %d) for %ld elms.\n",
                block.x, block.y, block.z,
                grid.x,  grid.y,  grid.z, pfb.nSmps);

    //Convert to floating point
    cu_unquantize<<<grid, block>>>(pfb.buf+pfb.nFir, pfb.bitty, pfb.nSmps);

    convolve<<<grid, block>>>(pfb.smps, pfb.buf+pfb.nFir, pfb.fir, pfb.nFir,
            pfb.nSmps, pfb.nChan);

    //Post Process
    apply_fft(pfb.buf, pfb.buf, pfb);

    if(hack) checked(cudaMemcpyAsync(hack, pfb.buf, pfb.nSmps*sizeof(float), cudaMemcpyDeviceToHost));

    //Convert to fixed point
    cu_quantize<<<grid, block>>>(pfb.bitty, pfb.buf, pfb.nSmps, pfb.nChan);

    //Retreive
    checked(cudaMemcpyAsync(buf, pfb.bitty, pfb.nSmps, cudaMemcpyDeviceToHost));

    //TODO copy back all information or discard unwanted portions, as they are
    //not needed
}

void apply_pfb_direct(int8_t *buffer, Pfb *p)
{
    p->run((uint8_t*)buffer);
}

void apply_pfb(float *buffer, Pfb *p)
{
    const size_t N = p->nSmps;
    uint8_t *buf   = p->h_buf;
    apply_quantize(buf, buffer, N);
    p->run(buf);
    apply_unquantize(buffer, buf, N);

    //rescale w/ fudge factor
    //for(size_t i=0; i<N; ++i)
    //    buffer[i] *= chans;
}
