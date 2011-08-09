#include <cuda.h>
#include <math_functions.h>
#include <cufft.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include "siggen.h"
#include "process.h"
#include <err.h>
#include <iostream>

//Function Declarations
void apply_polyphase(int8_t *buf, Pfb &pfb, float *hack=NULL);

//Macro Definitions
#define checked(x) { cudaError_t e = x; if(e!= cudaSuccess) {\
    fprintf(stderr,"CUDA error[%s][%d]: %s\n", #x, __LINE__, cudaGetErrorString(e));\
    exit(-1);}}

#define check_launch { cudaError_t e = cudaGetLastError(); if(e!= cudaSuccess) {\
    fprintf(stderr,"CUDA launch error[%d]: %s\n", __LINE__, cudaGetErrorString(e));\
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
    void run(int8_t *data, float *hack=NULL) {
        apply_polyphase(data, *this, hack);
    };

    void sync(void) {
        checked(cudaStreamSynchronize(stream));
    }

    //Parameter sizes
    size_t nFir, nSmps, nChan;

    //Device Buffers
    int8_t *bitty;
    float   *fir,
            *buf,
            *smps;

    //Host buffer
    int8_t *h_buf;

    //FFT handle
    cufftHandle plan;

    //stream handle for execution flow
    cudaStream_t stream;
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
    checked(cudaStreamCreate(&stream));
    checked(cudaMemcpyAsync(fir, _fir, nFir*sizeof(float),
                cudaMemcpyHostToDevice, stream));

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

    fftchecked(cufftSetStream(plan, stream));
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

    cudaStreamDestroy(stream);
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

    checked(cudaMemcpy2DAsync(dest, (width+2)*sFloat, src, width*sFloat,
                width*sFloat, height, cudaMemcpyDeviceToDevice, pfb.stream));
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

__global__ void cu_quantize(int8_t *dest, const float *src, size_t N, size_t
        chans)
{
    LOC;
    if(i<N)
        dest[i] = (static_cast<int8_t>(src[i]*7.5f)&0xc0);//0x00);//src[i]/32);//*128.0);
}

__global__ void cu_unquantize(float *dest, const int8_t *src, size_t N)
{
    LOC;
    if(i<N)
        dest[i] = src[i];
}

//half compression function
__device__ inline int8_t hcomp(int32_t d)
{
    const int8_t *x = reinterpret_cast<const int8_t*>(&d);
    return (x[3]&0xc0)|((x[2]&0xc0)>>2)|((x[1]&0xc0)>>4)|((x[0]&0xc0)>>6);
}

__global__ void cu_compress(int8_t *dest, const int32_t *src, size_t N)
{
    LOC;
    if(i<N)
        dest[i] = hcomp(src[i]);
}

__global__ void cu_stripper(int8_t *dest, const int8_t *src, size_t destWidth, size_t bytes)
{
    LOC;
    if(i>bytes)
        return;

    size_t srcWidth = destWidth+4;
    //transform source
    size_t j = i%destWidth + (i/destWidth)*srcWidth;
    dest[i] = src[j];
}

__global__ void convolve(float *dest, const float *src, const float *coeff,
        size_t nC, size_t nS, size_t chan)
{
    LOC;
    if (i<nS) {
        nC = 256;
        unsigned     sel   = i%chan;

        //do actual work at i
        float result = 0.0f;
#pragma unroll
        for(size_t j=sel; j<nC; j+=chan)
            result += src[i-sel-j]*coeff[j];
        dest[i] = result/chan;
    }
}

void apply_polyphase(int8_t *buf, Pfb &pfb, float *hack)
{
    checked(cudaMemcpyAsync(pfb.bitty, buf, pfb.nSmps, cudaMemcpyHostToDevice, pfb.stream));
    //Buffer with zeros
    checked(cudaMemsetAsync(pfb.buf, 0, pfb.nFir*sizeof(float), pfb.stream));

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
    cu_unquantize<<<grid, block, 0, pfb.stream>>>(pfb.buf+pfb.nFir, pfb.bitty, pfb.nSmps);
    check_launch;

    convolve<<<grid, block, 0, pfb.stream>>>(pfb.smps, pfb.buf+pfb.nFir, pfb.fir, pfb.nFir,
            pfb.nSmps, pfb.nChan);
    check_launch;

    //Post Process
    apply_fft(pfb.buf, pfb.buf, pfb);
    check_launch;

    //Provide all channels with full precision
    if(hack){
        checked(cudaMemcpyAsync(hack, pfb.buf, pfb.nSmps*sizeof(float), cudaMemcpyDeviceToHost, pfb.stream));
        checked(cudaStreamSynchronize(pfb.stream));
    }

    //Convert to fixed point
    cu_quantize<<<grid, block, 0, pfb.stream>>>(pfb.bitty, pfb.buf, pfb.nSmps, pfb.nChan);
    check_launch;

    //Retreive
#if PFB_OUTPUT_PACKED
    size_t width = pfb.nChan - 2;


    //checked(cudaMemsetAsync(pfb.smps, 0, pfb.nSmps, pfb.stream));

    //no not that kind
    cu_stripper<<<grid, block, 0, pfb.stream>>>((int8_t *)pfb.smps,
            pfb.bitty+2, width, pfb.nSmps);
    check_launch;

    checked(cudaMemsetAsync(pfb.bitty, 0xdb, pfb.nSmps, pfb.stream));

    cu_compress<<<grid, block, 0, pfb.stream>>>(pfb.bitty, (const
                int32_t*)pfb.smps, pfb.nSmps/4);
    check_launch;

    size_t lenScaled = pfb.nSmps*(pfb.nChan+2)/pfb.nChan/4;
    checked(cudaMemcpyAsync(buf, pfb.bitty, lenScaled, cudaMemcpyDeviceToHost, pfb.stream));
#else
    cu_compress<<<grid, block, 0, pfb.stream>>>((int8_t*)pfb.smps, (const
                int32_t*)pfb.bitty, pfb.nSmps/4);
    check_launch;

    size_t lenScaled = pfb.nSmps*(pfb.nChan+2)/pfb.nChan/4;
    checked(cudaMemcpyAsync(buf, pfb.smps, lenScaled, cudaMemcpyDeviceToHost, pfb.stream));
#endif

}

void apply_pfb_direct(int8_t *buffer, Pfb *p)
{
    p->run((int8_t*)buffer);
}

void sync_pfb_direct(Pfb *p)
{
    p->sync();
}

void apply_pfb(float *buffer, Pfb *p)
{
    const size_t N = p->nSmps;
    int8_t *buf   = p->h_buf;
    apply_quantize(buf, buffer, N);
    p->run(buf, buffer);
    p->sync();
}

void *getBuffer(size_t N)
{
    void *tmp;
    checked(cudaMallocHost(&tmp, N));
    return tmp;
}

void freeBuffer(void *b)
{
    if(b)
        checked(cudaFreeHost(b));
}

static int8_t getPt(const int8_t *data, off_t N)
{
    size_t address = N/4;
    size_t shift   = 2*(N%4);
    return (data[address]>>shift)&0x3;
}

complex<float> getSmp(const int8_t *data, off_t N)
{
    return complex<float>(float(getPt(data, 2*N)),
                          float(getPt(data, 2*N+1)));
}
