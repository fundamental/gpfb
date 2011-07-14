#include<cxxtest/TestSuite.h>
#include "../process.h"
#include "../siggen.h"
#include "../param.h"
#include "../packet.h"
#include "../rdbe.h"
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <algorithm>


//Verify that Realtime execution is possible [takes a while to run]
class realtimeTest : public CxxTest::TestSuite
{
    public:
    //Run PFB a large number of times to see appx data rate
    void testSpeed(void)
    {
        const size_t       TEST_LENGTH = 1<<10;
        unsigned long long processed   = 0;

        //Generate FIR coeffs
        float fir[TAPS];
        gen_fir(fir, TAPS, CHANNELS);
        scale_fir(fir, TAPS);
        window_fir(fir, TAPS);

        size_t DataSize = VDIFF_SIZE-sizeof(vheader_t),
               chunk    = (1<<13),
               length   = chunk*DataSize;

        //create working and loading copies
        class Pfb *l_pfb = alloc_pfb(fir, TAPS, length, CHANNELS);
        class Pfb *w_pfb = alloc_pfb(fir, TAPS, length, CHANNELS);
        int8_t *loading = (int8_t *) getBuffer(length+sizeof(vheader_t));
        int8_t *working = (int8_t *) getBuffer(length+sizeof(vheader_t));

        //Execute timed pfb
        struct timeval begin, end;
        gettimeofday(&begin, NULL);
        rdbe_connect();
        for(size_t i=0; i<TEST_LENGTH; ++i, processed += length) {
            //Gen signal
            sync_pfb_direct(l_pfb);
            //rdbe_gather(chunk, loading);

            std::swap(loading, working);
            std::swap(l_pfb, w_pfb);
            //Filter
            apply_pfb_direct(working+sizeof(vheader_t), w_pfb);
            putchar('.');
            fflush(stdout);
        }
        
        gettimeofday(&end, NULL);
        struct timeval dt;
        timersub(&end, &begin, &dt);
        double time_spent = (double)(dt.tv_sec*1.0+dt.tv_usec*1e-6);
        double gsps = processed/time_spent/1e9;
        puts("");
        printf("Lost packets:      %lu of %lu\n", missed_packet(), (chunk*TEST_LENGTH));
        printf("Drop rate:         %f%%\n", missed_packet()*100.0/(chunk*TEST_LENGTH));
        printf("Chunk size:        %fMB\n", length/1e6);
        printf("Samples processed: %lld\n", processed);
        printf("Size processed:    %fGB\n", processed/1e9);
        printf("Time spent:        %f\n", time_spent);
        printf("Giga-Samples/sec:  %f\n", gsps);
        
        rdbe_disconnect();
        delete_pfb(l_pfb);
        delete_pfb(w_pfb);
        freeBuffer(loading);
        freeBuffer(working);
    }
};

