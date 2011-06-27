#include<cxxtest/TestSuite.h>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include "../process.h"
#include "../siggen.h"
#include "../param.h"

static float to_real(float x, float y)
{
    return sqrt(x*x+y*y);
};

static float sqr(float x)
{
    return x*x;
}

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

class spectrumTest : public CxxTest::TestSuite
{
    public:
    /*Verify Bin Sepatation is at least 40dB in magnitude from center
     *frequencies */
    void testBins(void)
    {
        data = new float[MEM_SIZE];
        //Assuming all parameters should be used for testing
        float stepSize = FS*0.01/CHANNELS;
        float chans[CHANNELS/2+1];

        for(unsigned i=0; i*stepSize<FS/2.0; ++i) {
            const float freq = i*stepSize;
            //printf("#frequency:=%f",freq);
            avg(genCosResponse(freq), chans);
            for(size_t i=0;i<=CHANNELS/2;++i)
                printf("%c%f",i?',':'\n',chans[i]);
        }
        delete[] data;
    }

    private:

    float *genCosResponse(float frequency)
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
        apply_pfb(data, MEM_SIZE, fir, TAPS, CHANNELS);

        return data;
    }

    float *data;
};

