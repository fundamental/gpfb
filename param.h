#include <stddef.h>

//Size of a frame of data
const unsigned FRAMES = 64;

//Sample Rate
const unsigned FS = 1024;//MHz

//Filter Parameters
const size_t CHANNELS = 32,
             TAPS     = CHANNELS*8,
             MEM_SIZE = TAPS*FRAMES;

//Numeric Constants
const double PI = 3.14159265358979323846;
