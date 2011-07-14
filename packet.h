#include <cstddef>
#include <stdint.h>

namespace packet
{
    const size_t VDIFF_SIZE = 8224;//from vdiff documentation
    typedef uint32_t vheader_t[8]; //size of packet header

    //Total lost packets
    size_t missed(void);

    //Reset count of missed packets
    void resetMissed(void);

    //Verifies contents of header
    void checkHeader(vheader_t head);

    //Read in packet data
    void process(int8_t *out, const int8_t *in);

    //Print contents of the packet buffer
    void print(const int8_t *data, size_t N);
};
