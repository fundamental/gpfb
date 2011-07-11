#include<cxxtest/TestSuite.h>
#include "../process.h"
#include "../siggen.h"
#include "../param.h"
#include "../packet.h"
#include "../rdbe.h"
#include <string.h>
#include <stdio.h>
#include <time.h>


//Verify that Realtime execution is possible [takes a while to run]
class realtimeTest : public CxxTest::TestSuite
{
    public:
    //Run PFB a large number of times to see appx data rate
    void testSpeed(void)
    {
        const size_t       TEST_LENGTH = 128;
        unsigned long long processed   = 0;

        //Generate FIR coeffs
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);
        window_fir(fir, TAPS);

        size_t DataSize = VDIFF_SIZE-sizeof(vheader_t),
               length   = TAPS*DataSize;

        class Pfb *pfb = alloc_pfb(fir, TAPS, length, CHANNELS);
        rdbe_connect();

        //Execute timed pfb
        const clock_t begin = clock();
        for(size_t i=0; i<TEST_LENGTH; ++i, processed += length) {
            //Gen signal
            data = (int8_t *) rdbe_gather(TAPS);

            //Filter
            apply_pfb_direct(data, pfb);
            putchar('.');
            fflush(stdout);
        }
        
        double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;;
        double gsps = processed/time_spent/1e9;
        puts("");
        printf("Chunk size: %fMB\n", MEM_SIZE/1e6);
        printf("Time spent: %f\n", time_spent);
        printf("Samples processed: %lld\n", processed);
        printf("Giga-Samples per second: %f\n", gsps);
        printf("Times Real Time %f\n", gsps*8.0/10);
        
        rdbe_disconnect();
        delete_pfb(pfb);
        rdbe_free(data);
    }

    private:
    int8_t *data;
};

