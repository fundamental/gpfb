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

static void square_average(const float *data, float *chans)//[CHANNELS/2+1]
{
    memset(chans, 0, (CHANNELS/2+1)*sizeof(float));
    float2 *out = (float2*)data;
    for(size_t rowidx=0,i=0;i<MEM_SIZE/2;i++,rowidx++) {
        float smp = to_real(out[i].x, out[i].y);
        rowidx %= (CHANNELS/2+1);
        chans[rowidx] += smp*smp;
    }

    for(size_t i=0;i<=CHANNELS/2;++i)
        chans[i] = chans[i]*MEM_SIZE/(CHANNELS+2);
}

class freqTest : public CxxTest::TestSuite
{
    public:
    /*Verify Bin Sepatation is at least 40dB in magnitude from center
     *frequencies */
    void testBins(void)
    {
        //Assuming all parameters should be used for testing
        float stepSize = FS*1.0/CHANNELS;

        //Require at least 40dB channel separation
        for(unsigned i=0; i<CHANNELS/2+1; ++i) {

            const float freq = i*stepSize;
            float chans[CHANNELS/2+1];
            square_average(genCosResponse(freq), chans);

            size_t expect = i;//==CHANNELS/2 ? 0 : i+1;

#if 0
            std::cout << "frequency = " << freq << std::endl;
            //Expected location of signal
            for(int j=0; j<CHANNELS/2+1; ++j) {
                std::cout << "chan[" << j << "] = " << chans[j];
                if(j==expect) std::cout << '*' << std::endl;
                else std::cout << std::endl;
            }
#endif

            //Ensure the selected channel has a signal
            TS_ASSERT_LESS_THAN(1.0f, chans[expect]);
            const float thresh = chans[expect]/100.0f;

            for(unsigned j=0; j<CHANNELS/2+1; ++j)
                if(j!=expect)
                    TS_ASSERT_LESS_THAN(chans[j], thresh);
        }
    }

    private:

    float *genCosResponse(float frequency)
    {
        //Zero Out memory
        memset(data, 0, sizeof(data));

        //Generate Cosine
        gen_cos(data+TAPS, MEM_SIZE, frequency);
        
        //Generate FIR coeffs [could have scope expanded]
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);

        //Execute pfb
        apply_pfb(data+TAPS, MEM_SIZE, fir, TAPS, CHANNELS);

        return data+TAPS;
    }

    float data[TAPS+MEM_SIZE];
};

