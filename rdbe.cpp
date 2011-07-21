#include <err.h>    //for error display+exit
#include <assert.h> //for runtime checks
#include <stdlib.h>  //for memory access

#include <unistd.h>  //for sockets
#include <sys/socket.h>
#include <arpa/inet.h>

#include "rdbe.h"
#include "packet.h"

using packet::VDIFF_SIZE;

/**
 * RDBE Parameters - aka global constants
 * TODO move to a config file when possible (see libconfig)
 */
static const char *Addr = "192.168.5.102";
static const int   Port = 4201;

static int sock = -1;
//Get packet and place on queue
static ssize_t receive(void *buf, size_t len)
{
    int nb = recv(sock, buf, len, 0);
    if(nb == 0)
        warn("It appears that the rdbe has closed the connection");
    if(nb < 0)
        err(1, "Failed to get rdbe packet");
    if(nb != (int) VDIFF_SIZE)
        warnx("Packet size does not match VDIFF_SIZE %d != %d", nb,
                (int) VDIFF_SIZE);
    return nb;
}

void rdbe::connect(void)
{
    // preparation for bind
    struct sockaddr_in dname;
    dname.sin_family      = AF_INET;
    dname.sin_port        = htons(Port);
    dname.sin_addr.s_addr = inet_addr(Addr);

    if(dname.sin_addr.s_addr == INADDR_NONE)
        err(1, "Address '%s' is unintelligible.", Addr);

    // get a socket
    sock = socket(PF_INET, SOCK_DGRAM, 0);
    if(sock < 0) err(1, "Socket failure");

    // bind the socket to the address
    if(bind(sock, (struct sockaddr *)&dname, sizeof(dname)))
        err(1, "Failed to bind to port");
}

void rdbe::disconnect(void)
{
    close(sock);
    sock = -1;

    if(packet::missed())
        warn("A total of %lu packets were dropped\n", packet::missed());
}

//faster than a memcpy
static void preserve(int64_t *p, const int64_t *m)
{p[0]=m[0];p[1]=m[1];p[2]=m[2];p[3]=m[3];}
static void restore(int64_t *m, const int64_t *p)
{m[0]=p[0];m[1]=p[1];m[2]=p[2];m[3]=p[3];}

void rdbe::gather(int8_t *memory, size_t Packets)
{
    //define locals
    assert(sizeof(packet::vheader_t)*8/64==4);
    int64_t p[4];

    //Verify preconditions
    assert(memory);

    if (sock<0) errx(1, "%s called without rdbe_connect()", __func__);

    //collect packets
    for(size_t i=0; i<Packets; ++i) {
        const off_t offset = i*(VDIFF_SIZE-sizeof(packet::vheader_t));
        preserve(p, reinterpret_cast<const int64_t*>(memory+offset));
        receive(memory+offset, VDIFF_SIZE);
        packet::checkHeader(reinterpret_cast<const uint32_t *>(memory+offset));
        restore(reinterpret_cast<int64_t*>(memory+offset), p);
    }
}

