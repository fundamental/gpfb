
/* Put fir coeffs into buf with length taps
 * The filter has a corner frequency at fc (normalized)
 */
float *gen_fir(float *buf, unsigned taps, size_t chans);

//Rescale FIR filter coeff
float *scale_fir(float *buf, unsigned N);

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

//TODO update function to new conventions
float *gen_chirp(float *buf, size_t N, size_t period, double dr);
