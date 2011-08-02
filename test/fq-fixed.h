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

//overwrite variable
#define MEM_SIZE (1<<14)
#if PFB_OUTPUT_PACKED
# define CH_OUT (CHANNELS/2-1)
#else
# define CH_OUT (CHANNELS/2+1)
#endif

void print_fn(const int8_t *data, size_t N=(MEM_SIZE/CHANNELS*(CHANNELS/2-2)))
{
#if 0
    size_t asdfasdf = 34;
    for(size_t i=0;i<N/2/asdfasdf;++i) {
        for(size_t j=0;j<asdfasdf;++j)
            printf(" %.2hx", data[asdfasdf*i+j]);
        puts("");
    }
#endif

    static size_t rowidx=0;
    for(size_t i=0;i<N;i++,rowidx++) {
        float smp = abs(getSmp(data, i));
        rowidx %= CH_OUT;
        printf("%c%f", rowidx?',':'\n', smp);
    }

    //account for currently missing samples
    printf("\n");
    rowidx = 0;
}

//warning a large amount of discards are required for fixed testing
static void average(int8_t *data, float *chans, size_t discards=8)
{
    int82 *out = reinterpret_cast<int82*>(data);
    size_t length = MEM_SIZE/CHANNELS*(CHANNELS/2-2);

    //print_fn(data);

    memset(chans, 0, CH_OUT*sizeof(float));
    //discard the first several rows
    memset(out, 0, CH_OUT*discards);
    //print_fn(data);
    for(size_t rowidx=0,i=0;i<length;i++,rowidx++) {
        rowidx %= CH_OUT;
        chans[rowidx] += abs(getSmp(data, i));
        //printf("%c%f", rowidx?',':'\n', abs(getSmp(data, i)));
    }

    printf("[");
    for(size_t i=0;i<CH_OUT;++i)
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
#if PFB_OUTPUT_PACKED
            const float freq = (ch+1)*chSize;
#else
            const float freq = ch*chSize;
#endif

            float chans[CHANNELS/2-1];
            average(genCosResponse(freq, data), chans);

            //Expected channel should have a signal
            TS_ASSERT_LESS_THAN(100, chans[ch]);

            const float thresh = chans[ch]/minSep;

            float chmax = 0.0f;
            int   mchan = -1;
            //Ignore the DC and aliased channels
#if !PFB_OUTPUT_PACKED
# define extra_chan 2
#else
# define extra_chan 0
#endif
            for(unsigned j=0; j<CHANNELS/2+extra_chan; ++j) {
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
