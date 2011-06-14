#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <err.h>

const double PI = 3.14159265358979323846;

float sinc(float x)
{
    if (x == 0.0f)
        return 1.0;
    else
        return sin(x)/x;
}

float *gen_fir(float *buf, unsigned taps, float fc)
{
    assert(!(taps%2));

    for(size_t i=0; i<taps; ++i)
        buf[i] = sinc(PI*fc*(i-taps/2.0))*fc;
    return buf;
}

float *gen_rand(float *buf, size_t N, float norm)
{
    for(size_t i=0;i<N;++i)
        buf[i]=rand()*norm/RAND_MAX - norm/2.0;
    return buf;
}

float *gen_saw(float *buf, size_t N, size_t period)
{
    const float low = period/-2.0;
    printf("%f\n", low);
    float state     = low;
    for(size_t i=0;i<N;++i)
        state = buf[i] = i%period ? state+1.0 : low;
    return buf;
}

__global__ void convolve(float *coeff, size_t N, float *src, size_t M, float *dest, size_t chans)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>M) 
        return;

    //do actual work at i
    dest[i] = 0.0;
    for(size_t j=0; j<N; ++j)
        if(j%chans==0)
            dest[i] += src[i-j]*coeff[j];
}

float *apply_fir(float *buf, size_t N, float *coeff, size_t M, size_t chans)
{
    //insure clean decimation
    assert(M%chans==0);
    assert(N%chans==0);

    float *cu_coeff=NULL, *cu_buf=NULL, *cu_smps=NULL;//r,r,w

    //Allocate
    puts("Allocating...");
    cudaMalloc((void **)&cu_coeff, M*sizeof(float));
    cudaMalloc((void **)&cu_smps, N*sizeof(float));
    cudaMalloc((void **)&cu_buf, (N+M)*sizeof(float));

    //Send
    puts("Sending...");
    cudaMemcpy(cu_coeff, coeff, M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_buf, buf-M, (N+M)*sizeof(float), cudaMemcpyHostToDevice);

    //Run
    puts("Running...");
    int block_size = 128;
    int blocks = N/block_size + (N%block_size == 0 ? 0:1);
    convolve <<< blocks, block_size >>>(cu_coeff, M, cu_buf+M, N, cu_smps, chans);

    //Retreive
    puts("Getting...");
    cudaMemcpy(buf, cu_smps, sizeof(float)*N, cudaMemcpyDeviceToHost);

    //Clean
    puts("Cleaning...");
    cudaFree(cu_coeff);
    cudaFree(cu_smps);
    cudaFree(cu_buf);
    return buf;

}

int main()
{
    const size_t CHANNELS = 4,
                 N        = CHANNELS*8;
    float fir[N];
    gen_fir(fir, N, 1.0/CHANNELS);

    const size_t M=N*128;
    float buf[M+N];
    memset(buf, 0, N*sizeof(float));
    float *smps=buf+N;
    //gen_rand(smps, M, 1.0);
    gen_saw(smps, M, 1024);

    //Show previous
    FILE *fb = fopen("before.txt", "w+");
    for(size_t i=0;i<M;++i)
        fprintf(fb, "%f, ", smps[i]);
    fclose(fb);

    //Apply to samples
    apply_fir(smps, M, fir, N, CHANNELS);

    //Show results
    FILE *fa = fopen("after.txt", "w+");
    for(size_t i=0;i<M;++i)
        fprintf(fa, "%f, ", smps[i]);
    fclose(fa);
    return 0;
}
