#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static const size_t VDIFF_SIZE = 8224;//from vdiff documentation
typedef uint32_t vheader_t[8]; //size of packet header

//total lost packets
size_t missed_packet(void);

//Read in packet data
void process_packet(int8_t *out, const int8_t *in);

//Print contents of the packet buffer
void print_packets(const int8_t *data, size_t N);

#ifdef __cplusplus
}
#endif
