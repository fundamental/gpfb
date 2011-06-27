#include<cxxtest/TestSuite.h>
#include "../process.h"
#include "../siggen.h"
#include "../param.h"
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


        data = new float[MEM_SIZE];
        //pfb_t *pfb = alloc_pfb(fir, TAPS, MEM_SIZE, CHANNELS);

        //Execute timed pfb
        const clock_t begin = clock();
        for(size_t i=0; i<TEST_LENGTH; ++i, processed += MEM_SIZE) {
            //Gen signal
            //memset(data, 0, sizeof(data));
            //gen_rand(data, MEM_SIZE, 2.0);

            //Filter
            //apply_pfb(data, pfb);

            apply_pfb(data, MEM_SIZE, fir, TAPS, CHANNELS);
            putchar('.');
            fflush(stdout);
        }
        
        double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;;
        double gsps = processed/time_spent/1e9;
        puts("");
        printf("Chunk size: %fMB\n", MEM_SIZE/1e6);
        printf("Time spent: %f\n", time_spent);
        printf("Samples processed: %ld\n", processed);
        printf("Giga-Samples per second: %f\n", gsps);
        printf("Times Real Time %f\n", gsps*8.0/10);
        
        //delete_pfb(pfb);
        delete[] data;
    }

    private:
    float *data;
};

