#include <stdio.h>
#include <stdint.h>
#include <err.h>
#include <errno.h>
#include <stdlib.h>
#include "packet.h"
#include "rdbe.h"


#ifdef VERBOSE
#define comment(...) fprintf(stderr, __VA_ARGS__)
#else
#define comment(...)
#endif

#define ch(x) { int ret = x; if(ret) err(ret, "Failed to exec %s", #x);}
int fsize(FILE *f)
{
    ch(fseek(f, 0, SEEK_END));// seek to end of file
    int size = ftell(f);
    ch(fseek(f, 0, SEEK_SET));//return to start
    return size;
}

int main(int argc, char **argv)
{
    if(argc != 2) errx(1, "usage %s #packets", *argv);

    const size_t Packets   = atoll(argv[1]),
                 Vpad      = sizeof(packet::vheader_t),
                 FrameSize = packet::VDIFF_SIZE-Vpad,
                 DataSize  = FrameSize*Packets;

    comment("#Packet Count: %lu\n", Packets);
    comment("#Buffer Size:  %lu\n", DataSize);

    int8_t *data = new int8_t[DataSize+Vpad];
    rdbe::connect();
    rdbe::gather(data, Packets);
    rdbe::disconnect();

    packet::print(data, DataSize);

    delete [] data;

    return 0;
}

