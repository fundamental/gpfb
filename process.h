#include <stddef.h>

//Result is returned interleaved in buffer
void apply_pfb(float *buffer, size_t N, float *coeff, size_t taps, size_t chans);
