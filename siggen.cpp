#include "param.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "siggen.h"

static float sinc(float x)
{
    if (x == 0.0f)
        return 1.0;
    else
        return sin(x)/x;
}


float *gen_fir(float *buf, unsigned taps, size_t chans)
{
    assert(!(taps%2));
    assert(chans>1);
    float fc = 1.0/chans;

    for(size_t i=0; i<taps; ++i)
        buf[i] = sinc(PI*fc*(i-taps/2.0))*fc;
    return buf;
}

float *scale_fir(float *buf, unsigned N)
{
    float max = 0.0;
    for(size_t i=0; i<N; ++i)
        max = buf[i]>max?buf[i]:max;

    for(size_t i=0; i<N; ++i)
        buf[i] /= max;
    return buf;
}

// hamming at n of N
static float hamming(size_t n, size_t N)
{
    return 0.54-0.46*cos(2*PI*n/N);
}

float *window_fir(float *buf, size_t N)
{
    for(size_t i=0;i<N;++i)
        buf[i] *= hamming(i,N);
}

void apply_quantize(uint8_t *dest, const float *src, size_t N)
{
    for(size_t i=0;i<N;++i)
        dest[i] = quantize(src[i]);
}

void apply_unquantize(float *dest, const uint8_t *src, size_t N)
{
    for(size_t i=0;i<N;++i)
        dest[i] = unquantize(src[i]);
}

float *gen_rand(float *buf, size_t N, float norm)
{
    for(size_t i=0;i<N;++i)
        buf[i] += rand()*norm/RAND_MAX - norm/2.0;
    return buf;
}

float *gen_imp(float *buf, size_t N)
{
    (void) N;
    *buf += 1.0;
    return buf;
}

float *gen_step(float *buf, size_t N)
{
    for(size_t i=0;i<N;++i)
        buf[i] += 1.0;
    return buf;
}

float *gen_saw(float *buf, size_t N, float fq)
{
    const int period = FS/fq;
    const float low = period/-2.0;
    float state     = low;
    for(size_t i=0;i<N;++i)
        buf[i] += (state = i%period ? state+1.0 : low)/-low;
    return buf;
}

float *gen_dc(float *buf, size_t N)
{
    for(size_t i=0;i<N;++i)
        buf[i] += 1.0;
    return buf;
}

float *gen_cos(float *buf, size_t N, float fq)
{
    const float rate = 2.0*PI*fq/FS;
    for(size_t i=0;i<N;++i)
        buf[i] += cos(rate*i);
    return buf;
}

float *gen_chirp(float *buf, size_t N, size_t period, double dr)
{
    double rate = 2.0*PI/period,
           state = 0;
    for(size_t i=0;i<N;++i,state+=rate,rate+=dr)
        buf[i] += sin(state);
    return buf;
}
