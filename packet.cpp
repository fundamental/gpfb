#include "packet.h"
#include <err.h>    //err & warn
#include <cassert>  //assert
#include <string.h> //memcpy
#include <cstdio>   //printf

#define ch(x) { int ret = x; if(ret) err(ret, "Failed to exec %s", #x);}

static size_t PacketSkipped = 0;
size_t packet::missed(void) {return PacketSkipped;}
void packet::resetMissed(void) {PacketSkipped = 0;}

static void skip_packet(size_t _skipped)
{
    PacketSkipped += _skipped;
    //warnx("skipped a chunk of %lu packets!", _skipped);
}

static void set_packet(int32_t pid)
{
    //comment("Packet #<%x>\n", pid);
    static int32_t previous = 0;
    if(previous && pid-previous != 1)
        skip_packet(pid-previous-1);
    previous = pid;
}

void packet::checkHeader(const packet::vheader_t head)
{
    set_packet(head[6]);
    if(head[4]||head[5]) errx(1, "Invalid Packet detected");
}

void packet::process(int8_t *out, const int8_t *in)
{
    //Gather packet header
    vheader_t head;
    memcpy(head, in, sizeof(vheader_t));
    in += sizeof(vheader_t);
    checkHeader(head);

    const size_t length = VDIFF_SIZE-sizeof(vheader_t);

    //Gather Packet Data
    int32_t *data = (int32_t*) out;
    memcpy(data, in, length);
}

void packet::print(const int8_t *data, size_t N)
{
    for(size_t i=0;i<N;++i)
        printf("%d\n", data[i]);
    
    if(PacketSkipped)
        warnx("total Missing Packets - %lu", PacketSkipped);
};
