#include <stddef.h>
#include <stdint.h>
#include <complex>

class Pfb;
//Result is returned interleaved in buffer
void apply_pfb(float *buffer, Pfb *p);
void apply_pfb_direct(int8_t *buffer, Pfb *p);
void sync_pfb_direct(Pfb *p);

class Pfb *alloc_pfb(const float *fir, size_t _nFir, size_t _nSmps, size_t _nChan);
void delete_pfb(class Pfb *p);

void *getBuffer(size_t N);
void freeBuffer(void *b);

using std::complex;
complex<float> getSmp(const int8_t *data, off_t N);
