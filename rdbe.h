namespace rdbe
{
    //Non-rentrant rdbe connection
    void connect(void);
    void disconnect(void);

    /**
     * Gather a set of packets `Packets' long
     * Store result into memory discarding headers
     * Will be length (VDIFF_SIZE-sizeof(vheader_t))*Packets
     */
    void gather(int8_t *memory, size_t Packets);
};
