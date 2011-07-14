#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>
#include <assert.h>
#include <algorithm>
#include "siggen.h"
#include "param.h"
#include "rdbe.h"
#include "packet.h"
#include "process.h"

struct cpx_t{uint8_t x,y;};

#define MHZ *1.0

//Complex to real
static float to_real(const cpx_t &cpx)
{
    const float x = unquantize(cpx.x),
                y = unquantize(cpx.y);

    return sqrt(x*x+y*y);
}

void execute_pfb(const float *fir, void(*print)(const cpx_t*,size_t))
{
    //Decide on parameters of execution
    const size_t DataSize = VDIFF_SIZE-sizeof(vheader_t),
                 chunk    = 1<<11,
                 iter     = 2,
                 len      = DataSize*chunk;

    //Generate handles to resources
    class Pfb *cpu_pfb = alloc_pfb(fir, TAPS, len, CHANNELS);
    class Pfb *gpu_pfb = alloc_pfb(fir, TAPS, len, CHANNELS);
    int8_t *cpu_buf = (int8_t *) getBuffer(DataSize*chunk+sizeof(vheader_t));
    int8_t *gpu_buf = (int8_t *) getBuffer(DataSize*chunk+sizeof(vheader_t));
    rdbe_connect();

#if 1
    void gen_fixed_cos(int8_t *buf, size_t N, float fq);

    //rdbe_gather(chunk, gpu_buf);
    //rdbe_gather(chunk, cpu_buf);
    //gen_fixed_cos(cpu_buf, DataSize*chunk, 30.0 MHZ);
    apply_pfb_direct(gpu_buf+sizeof(vheader_t), gpu_pfb);
    apply_pfb_direct(cpu_buf+sizeof(vheader_t), cpu_pfb);


#else
    //Process loop
    for(size_t i=0; i<iter; ++i) {
        fputc('p', stderr);
        fflush(stderr);

        //Gather Signal
        sync_pfb_direct(cpu_pfb);
        rdbe_gather(chunk, cpu_buf);

        //Update state
        std::swap(cpu_buf, gpu_buf);
        std::swap(cpu_pfb, gpu_pfb);

        fputc('f', stderr);
        fflush(stderr);

        //Filter
        apply_pfb_direct(gpu_buf+sizeof(vheader_t), gpu_pfb);

        //Show status
        fputc('b', stderr);
        fflush(stderr);
    }
#endif

    //Ensure valid data exists
    sync_pfb_direct(gpu_pfb);
    sync_pfb_direct(cpu_pfb);

    //Print data
    //print((const cpx_t*)(gpu_buf+sizeof(vheader_t)), DataSize*chunk/2);
    //print((const cpx_t*)(cpu_buf+sizeof(vheader_t)), DataSize*chunk/2);

    //Cleanup
    rdbe_disconnect();
    delete_pfb(cpu_pfb);
    delete_pfb(gpu_pfb);
    freeBuffer(cpu_buf);
    freeBuffer(gpu_buf);
}

FILE *output = fopen("after.txt", "w+");
void print_fn(const cpx_t *out, size_t N)
{
    static size_t rowidx=0;
    for(size_t i=0;i<N;i++,rowidx++) {
        float smp = to_real(out[i]);
        rowidx %= (CHANNELS/2+1);
        fprintf(output, "%c%f", rowidx?',':'\n', smp);
    }
}

int main()
{
    float fir[TAPS];
    gen_fir(fir, TAPS, CHANNELS);
    scale_fir(fir, TAPS);
    window_fir(fir, TAPS);

    if(!output) err(1, "Could not open output");

    //Apply to samples from NIC
    execute_pfb(fir, print_fn);

    //cleanup
    fclose(output);
    return 0;
}
