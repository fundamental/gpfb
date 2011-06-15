#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <err.h>

const double PI = 3.14159265358979323846;

const double FS = 1024;//MHz
const unsigned FRAMES = 50;

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


//Generate random noise normalized to (-norm..norm)/2
float *gen_rand(float *buf, size_t N, float norm)
{
    for(size_t i=0;i<N;++i)
        buf[i] += rand()*norm/RAND_MAX - norm/2.0;
    return buf;
}

//Generate impulse
float *gen_imp(float *buf, size_t N)
{
    (void) N;
    *buf += 1.0;
    return buf;
}

//Generate sawtooth wave with given period in samples
float *gen_saw(float *buf, size_t N, size_t period)
{
    const float low = period/-2.0;
    float state     = low;
    for(size_t i=0;i<N;++i)
        buf[i] += (state = i%period ? state+1.0 : low)/-low;
    return buf;
}

//Generate dc offset
float *gen_dc(float *buf, size_t N)
{
    for(size_t i=0;i<N;++i)
        buf[i] += 1.0;
    return buf;
}

//generate sin wave at frequency fq
float *gen_sin(float *buf, size_t N, float fq)
{
    const float rate = 2.0*PI*fq/FS;
    for(size_t i=0;i<N;++i) //TODO change sin to cos after testing
        buf[i] += cos(rate*i);
    return buf;
}

//TODO update function to new conventions
float *gen_chirp(float *buf, size_t N, size_t period, double dr)
{
    double rate = 2.0*PI/period,
           state = 0;
    for(size_t i=0;i<N;++i,state+=rate,rate+=dr)
        buf[i] += sin(state);
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

#undef checked
#define checked(x) { if(x!=cudaSuccess) err(1, #x);}
float *apply_fir(float *buf, size_t N, float *coeff, size_t M, size_t chans)
{
    //insure clean decimation
    assert(M%chans==0);
    assert(N%chans==0);

    float *cu_coeff=NULL, *cu_buf=NULL, *cu_smps=NULL;//r,r,w

    //Allocate
    puts("Allocating...");
    checked(cudaMalloc((void **)&cu_coeff, M*sizeof(float)));
    checked(cudaMalloc((void **)&cu_smps, N*sizeof(float)));
    checked(cudaMalloc((void **)&cu_buf, (N+M)*sizeof(float)));

    //Send
    puts("Sending...");
    checked(cudaMemcpy(cu_coeff, coeff, M*sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(cu_buf, buf-M, (N+M)*sizeof(float), cudaMemcpyHostToDevice));

    //Run
    puts("Running...");
    int block_size = 128;
    int blocks = N/block_size + (N%block_size == 0 ? 0:1);
    convolve <<< blocks, block_size >>>(cu_coeff, M, cu_buf+M, N, cu_smps, chans);

    //Retreive
    puts("Getting...");
    checked(cudaMemcpy(buf, cu_smps, sizeof(float)*N, cudaMemcpyDeviceToHost));

    //Clean
    puts("Cleaning...");
    checked(cudaFree(cu_coeff));
    checked(cudaFree(cu_smps));
    checked(cudaFree(cu_buf));
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
