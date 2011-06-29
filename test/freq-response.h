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

typedef struct{float x,y;} float2;

static void average(const float *data, float *chans)//[CHANNELS/2+1]
{
    memset(chans, 0, (CHANNELS/2+1)*sizeof(float));
    float2 *out = (float2*)data;
    for(size_t rowidx=0,i=0;i<MEM_SIZE/2;i++,rowidx++) {
        rowidx %= (CHANNELS/2+1);
        chans[rowidx] += to_real(out[i].x, out[i].y);
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
        data = new float[MEM_SIZE];
        //Assuming all parameters should be used for testing
        float stepSize = FS*1.0/CHANNELS;

        //Require some channel separation
        for(unsigned i=0; i<CHANNELS/2+1; ++i) {

            const float freq = i*stepSize;
            float chans[CHANNELS/2+1];
            average(genCosResponse(freq), chans);

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
            const float thresh = chans[expect]/10.0f;

            for(unsigned j=0; j<CHANNELS/2+1; ++j)
                if(j!=expect)
                    TS_ASSERT_LESS_THAN(chans[j], thresh);
            putchar('.'), fflush(stdout);
        }
        delete [] data;
    }

    private:

    float *genCosResponse(float frequency)
    {
        //Zero Out memory
        memset(data, 0, sizeof(float)*MEM_SIZE);

        //Generate Cosine
        gen_cos(data, MEM_SIZE, frequency);
        
        //Generate FIR coeffs [could have scope expanded]
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);

        //Execute pfb
        class Pfb *pfb = alloc_pfb(fir, TAPS, MEM_SIZE, CHANNELS);
        apply_pfb(data, pfb);
        delete_pfb(pfb);

        return data;
    }

    float *data;
};

