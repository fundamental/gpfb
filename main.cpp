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

using packet::vheader_t;
using packet::VDIFF_SIZE;

struct cpx_t{int8_t x,y;};

#define MHZ *1.0

//Complex to real
static float to_real(const cpx_t &cpx)
{
    float x = (cpx.x>>6),
          y = (cpx.y>>6);
    x+=0.5f;
    y+=0.5f;
    //return x;

    return sqrt(x*x+y*y);
}

void execute_pfb(const float *fir, void(*print)(const cpx_t*,size_t))
{
    //Decide on parameters of execution
    const size_t DataSize = VDIFF_SIZE-sizeof(vheader_t),
                 Packets  = 1<<5,
                 Iter     = 2,
                 DataLen      = DataSize*Packets;

    //Generate handles to resources
    class Pfb *cpu_pfb = alloc_pfb(fir, TAPS, DataLen, CHANNELS);
    class Pfb *gpu_pfb = alloc_pfb(fir, TAPS, DataLen, CHANNELS);
    int8_t *cpu_buf = (int8_t *) getBuffer(DataLen+sizeof(vheader_t));
    int8_t *gpu_buf = (int8_t *) getBuffer(DataLen+sizeof(vheader_t));
    rdbe::connect();

#if 1
    void gen_fixed_cos(int8_t *buf, size_t N, float fq);

    rdbe::gather(gpu_buf, Packets);
    rdbe::gather(cpu_buf, Packets);
    memset(cpu_buf, 0, Packets);
    //gen_fixed_cos(cpu_buf, DataLen, 24.0 MHZ);
    gen_rand(gpu_buf, DataLen);
    apply_pfb_direct(gpu_buf+sizeof(vheader_t), gpu_pfb);
    apply_pfb_direct(cpu_buf+sizeof(vheader_t), cpu_pfb);


#else
    //Process loop
    for(size_t i=0; i<Iter; ++i) {
        fputc('p', stderr);
        fflush(stderr);

        //Gather Signal
        sync_pfb_direct(cpu_pfb);
        rdbe_gather(Packets, cpu_buf);

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
    //sync_pfb_direct(cpu_pfb);

    //Print data
    size_t length = DataLen/CHANNELS*(CHANNELS/2-2);
    print((const cpx_t*)(gpu_buf+sizeof(vheader_t)), length);
    //print((const cpx_t*)(cpu_buf+sizeof(vheader_t)), DataLen/2);

    //Cleanup
    rdbe::disconnect();
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
        rowidx %= (CHANNELS/2-1);
        fprintf(output, "%c%f", rowidx?',':'\n', smp);
    }

    //account for currently missing samples
    fprintf(output, "\n");
    rowidx = 0;
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
