#include<cxxtest/TestSuite.h>
#include "../process.h"
#include "../siggen.h"
#include "../param.h"
#include "../packet.h"
#include "../rdbe.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>


//Verify that Realtime execution is possible [takes a while to run]
class realtimeTest : public CxxTest::TestSuite
{
    public:
    //Run PFB a large number of times to see appx data rate
    void testSpeed(void)
    {
        const size_t       TEST_LENGTH = 1024;
        unsigned long long processed   = 0;

        //Generate FIR coeffs
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);
        window_fir(fir, TAPS);

        size_t DataSize = VDIFF_SIZE-sizeof(vheader_t),
               chunk    = (1<<10),
               length   = chunk*DataSize;

        class Pfb *pfb = alloc_pfb(fir, TAPS, length, CHANNELS);
        rdbe_connect();
        int8_t *loading = (int8_t *) getBuffer(length+sizeof(vheader_t));
        int8_t *working = (int8_t *) getBuffer(length+sizeof(vheader_t));

        //Execute timed pfb
        const clock_t begin = clock();
        for(size_t i=0; i<TEST_LENGTH; ++i, processed += length) {
            //Gen signal
            rdbe_gather(chunk, loading);

            std::swap(loading, working);
            //Filter
            apply_pfb_direct(working+sizeof(vheader_t), pfb);
            putchar('.');
            fflush(stdout);
        }
        
        double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;
        double gsps = processed/time_spent/1e9;
        puts("");
        printf("Lost packets:      %llu\n", missed_packet());
        printf("Drop rate:         %f%%\n", missed_packet()*100.0/processed);
        printf("Chunk size:        %fMB\n", length/1e6);
        printf("Samples processed: %lld\n", processed);
        printf("Size processed:    %fGB\n", processed/1e9);
        printf("Time spent:        %f\n", time_spent);
        printf("Giga-Samples/sec:  %f\n", gsps);
        printf("Times Real Time    %f\n", gsps*8.0/10);
        
        rdbe_disconnect();
        delete_pfb(pfb);
        freeBuffer(loading);
        freeBuffer(working);
    }
};

