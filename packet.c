#include "packet.h"
#include <err.h> //err & warn
#include <assert.h> //assert
#include <netinet/in.h> //ntohl
#include <string.h> //memcpy

#define ch(x) { int ret = x; if(ret) err(ret, "Failed to exec %s", #x);}
static size_t PacketSkipped = 0;
size_t missed_packet(void) {return PacketSkipped;}

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

void process_header(vheader_t head)
{
    set_packet(head[6]);
    if(head[4]||head[5]) errx(1, "Invalid Packet detected");
}

void process_packet(int8_t *out, const int8_t *in)
{
    //Gather packet header
    vheader_t head;
    memcpy(head, in, sizeof(vheader_t));
    in += sizeof(vheader_t);
    process_header(head);

    const size_t length = VDIFF_SIZE-sizeof(vheader_t);

    //Gather Packet Data
    int32_t *data = (int32_t*) out;
    memcpy(data, in, length);
}

void print_packets(const int8_t *data, size_t N)
{
    for(size_t i=0;i<N;++i)
        printf("%d\n", data[i]);
    
    if(PacketSkipped)
        warnx("total Missing Packets - %lu", PacketSkipped);
};
