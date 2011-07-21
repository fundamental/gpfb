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

typedef struct{int8_t x,y;} int82;

static void average(const int8_t *data, float *chans)//[CHANNELS/2+1]
{
    size_t length = MEM_SIZE/CHANNELS*(CHANNELS/2-2);
    memset(chans, 0, (CHANNELS/2-1)*sizeof(float));
    int82 *out = (int82*)data;
    for(size_t rowidx=0,i=0;i<length;i++,rowidx++) {
        rowidx %= (CHANNELS/2-1);
        chans[rowidx] += to_real(float(out[i].x), float(out[i].y));
    }

    printf("[");
    for(size_t i=0;i<CHANNELS/2-1;++i)
        printf("%f ", chans[i]);
    printf("]\n");
}


template<class T>
T&max(T&a,T&b){return a>b?a:b;}
#define HZ

//performs the quantized frequency test
class freqQuantTest : public CxxTest::TestSuite
{
    public:
    /*Verify Bin Sepatation is at least 40dB in magnitude from center
     *frequencies */
    void testBins(void)
    {
        int8_t *data = new int8_t[MEM_SIZE];
        const float chSize = FS*1.0/CHANNELS HZ,
                    minSep   = 30.0f;

        //Verify seperation condition
        for(unsigned ch=0; ch<CHANNELS/2-1; ++ch) {
            const float freq = (ch+1)*chSize;

            float chans[CHANNELS/2-1];
            average(genCosResponse(freq, data), chans);

            //Expected channel should have a signal
            TS_ASSERT_LESS_THAN(100, chans[ch]);

            const float thresh = chans[ch]/minSep;

            float chmax = 0.0f;
            int   mchan = -1;
            //Ignore the DC and aliased channels
            for(unsigned j=0; j<CHANNELS/2; ++j) {
                if(j==ch) continue;

                TS_ASSERT_LESS_THAN(chans[j], thresh);
                if(chmax < chans[j])
                    mchan = j;
                chmax = max(chmax, chans[j]);
            }

            //display gain diff
            printf("[%f:%d]", chans[ch]*1.0/chmax, mchan);
            putchar('.'), fflush(stdout);
        }
        delete [] data;
    }

    private:

    int8_t *genCosResponse(float frequency, int8_t *data)
    {
        //Zero Out memory
        memset(data, 0, MEM_SIZE);

        //Generate Cosine
        gen_fixed_cos(data, MEM_SIZE, frequency);

        //Generate FIR coeffs [could have scope expanded]
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);

        //Execute pfb
        class Pfb *pfb = alloc_pfb(fir, TAPS, MEM_SIZE, CHANNELS);
        apply_pfb_direct(data, pfb);
        sync_pfb_direct(pfb);
        delete_pfb(pfb);

        return data;
    }
};
