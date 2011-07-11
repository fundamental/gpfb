#include <stddef.h>
#include <stdint.h>

class Pfb;
//Result is returned interleaved in buffer
void apply_pfb(float *buffer, Pfb *p);
void apply_pfb_direct(int8_t *buffer, Pfb *p);

class Pfb *alloc_pfb(const float *fir, size_t _nFir, size_t _nSmps, size_t _nChan);
void delete_pfb(class Pfb *p);

void *getBuffer(size_t N);
void freeBuffer(void *b);
