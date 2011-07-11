#include <err.h>    //for error display+exit
#include <assert.h> //for runtime checks
#include <stdlib.h>  //for memory access

#include <unistd.h>  //for sockets
#include <sys/socket.h>
#include <arpa/inet.h>

#include "packet.h"

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
        warn("Packet size does not match VDIFF_SIZE %d != %d", nb, 
                (int) VDIFF_SIZE);
    return nb;
}

void rdbe_connect(void)
{
    // preparation for bind
    const struct sockaddr_in dname = {
        .sin_family = AF_INET,
        .sin_port   = htons(Port),
        .sin_addr   = { .s_addr = inet_addr(Addr)}
    };

    if(dname.sin_addr.s_addr == INADDR_NONE)
        err(1, "Address '%s' is unintelligible.", Addr);

    // get a socket
    sock = socket(PF_INET, SOCK_DGRAM, 0);
    if(sock < 0) err(1, "Socket failure");

    // bind the socket to the address
    if(bind(sock, (struct sockaddr *)&dname, sizeof(dname)))
        err(1, "Failed to bind to port");
}

void rdbe_disconnect(void)
{
    close(sock);
    sock = -1;
}

void rdbe_free(void *m)
{
    if(m)
        free(m-sizeof(vheader_t));
}

void rdbe_gather(size_t Packets, int8_t *memory)
{
    //define locals
    uint64_t p[4];//sizeof(vheader_t)/64
    void preserve(const uint64_t *m){p[0]=m[0];p[1]=m[1];p[2]=m[2];p[3]=m[3];}
    void restore(uint64_t *m){m[0]=p[0];m[1]=p[1];m[2]=p[2];m[3]=p[3];}

    assert(memory);

    if (sock<0) errx(1, "%s called without rdbe_connect()", __func__);

    //collect packets
    for(size_t i=0; i<Packets; ++i) {
        const off_t offset = i*(VDIFF_SIZE-sizeof(vheader_t));
        preserve((uint64_t*)(memory+offset));
        receive(memory+offset, VDIFF_SIZE);
        process_header(memory+offset);
        restore((uint64_t*)(memory+offset));
    }
}

