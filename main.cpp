#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>
#include <assert.h>
#include "siggen.h"
#include "param.h"
#include "process.h"

//Complex to real
static float to_real(float x, float y)
{
    return sqrt(x*x+y*y);
}

typedef struct{float x,y;} float2;

int main()
{
    float fir[TAPS];
    gen_fir(fir, TAPS, CHANNELS);
    scale_fir(fir, TAPS);
    window_fir(fir, TAPS);

    float *smps = new float[MEM_SIZE];
    assert(smps);
    memset(smps, 0, MEM_SIZE*sizeof(float));
    gen_chirp(smps, MEM_SIZE, 1024*16, 0.0008);
    gen_cos(smps, FRAMES, 500.0);
    //gen_dc(smps, MEM_SIZE);

    //Show previous
    FILE *fb = fopen("before.txt", "w+");
    if(!fb) err(1, 0, "Could not open output");
    for(size_t i=0;i<MEM_SIZE;++i)
        fprintf(fb, "%f, ", smps[i]);
    fclose(fb);

    //Apply to samples
    apply_pfb(smps, MEM_SIZE, fir, TAPS, CHANNELS);

    //Show results
    FILE *fa = fopen("after.txt", "w+");
    if(!fa) err(1, 0, "Could not open output");
    float2 *out = (float2*)smps;
    for(size_t rowidx=0,i=0;i<MEM_SIZE/2;i++,rowidx++) {
        float smp = to_real(out[i].x, out[i].y);
        rowidx %= (CHANNELS/2+1);
        fprintf(fa, "%c%f", rowidx?',':'\n', smp);
    }
    fclose(fa);

    delete[] smps;
    return 0;
}
