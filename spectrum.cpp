#include <string.h>
#include <cstdio>
#include <cmath>
#include "process.h"
#include "siggen.h"
#include "param.h"

static float to_real(float x, float y)
{
    return sqrt(x*x+y*y);
};

typedef struct{float x,y;} float2;

static void avg(const float *data, float *chans)//[CHANNELS/2+1]
{
    memset(chans, 0, (CHANNELS/2+1)*sizeof(float));
    float2 *out = (float2*)data;
    for(size_t rowidx=0,i=0;i<MEM_SIZE/2;i++,rowidx++) {
        float smp = to_real(out[i].x, out[i].y);
        rowidx %= (CHANNELS/2+1);
        chans[rowidx] += smp;
    }

    for(size_t i=0;i<=CHANNELS/2;++i)//Divide by length of avg and fft
        chans[i] /= CHANNELS*MEM_SIZE/(CHANNELS*2+4); //multiplication by 2 to finish norm
    //resulting values for pfb should be 0..1
}

float *genCosResponse(float *data, float frequency);
int main()
{
    float *data = new float[MEM_SIZE];
    //Assuming all parameters should be used for testing
    float stepSize = FS*0.01/CHANNELS;
    float chans[CHANNELS/2+1];

    for(unsigned i=0; i*stepSize<FS/2.0; ++i) {
        const float freq = i*stepSize;
        //printf("#frequency:=%f",freq);
        avg(genCosResponse(data, freq), chans);
        for(size_t i=0;i<=CHANNELS/2;++i)
            printf("%c%f",i?',':'\n',chans[i]);
    }
    delete[] data;
}


float *genCosResponse(float *data, float frequency)
{
    //Zero Out memory
    memset(data, 0, sizeof(MEM_SIZE)*sizeof(float));

    //Generate Cosine
    gen_cos(data, MEM_SIZE, frequency);
        
    //Generate FIR coeffs [could have scope expanded]
    float fir[TAPS];
    gen_fir(fir, TAPS, CHANNELS);
    scale_fir(fir, TAPS);
    window_fir(fir, TAPS);

    //Execute pfb
    class Pfb *pfb = alloc_pfb(fir, TAPS, MEM_SIZE, CHANNELS);
    apply_pfb(data, pfb);
    delete_pfb(pfb);

    return data;
}

