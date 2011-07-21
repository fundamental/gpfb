#include <stdint.h>

/* Put fir coeffs into buf with length taps
 * The filter has a corner frequency at fc (normalized)
 */
float *gen_fir(float *buf, unsigned taps, size_t chans);

//Rescale FIR filter coeff
float *scale_fir(float *buf, unsigned N);

//Window with the hamming window
float *window_fir(float *buf, size_t N);

//8-bit Quantization
/** 
 * Perform 8 bit quantization and dequantization 
 * Input is [-2.0..2.0], output is [0..255]
 */
const float q_factor = 127.0;
#define quantize(i) ((i))*q_factor
#define unquantize(i) (i)/q_factor
void apply_quantize(int8_t *dest, const float *src, size_t N);
void apply_unquantize(float *dest, const int8_t *src, size_t N);

//Generate random noise normalized to (-norm..norm)/2
float *gen_rand(float *buf, size_t N, float norm);

//Generate impulse
float *gen_imp(float *buf, size_t N);

//Generate step signal
float *gen_step(float *buf, size_t N);

//Generate sawtooth wave with given period in samples
float *gen_saw(float *buf, size_t N, float fq);

//Generate dc offset
float *gen_dc(float *buf, size_t N);

//generate cosine wave at frequency fq
float *gen_cos(float *buf, size_t N, float fq);
void gen_fixed_cos(int8_t *buf, size_t N, float fq);

//TODO update function to new conventions
float *gen_chirp(float *buf, size_t N, size_t period, double dr);
