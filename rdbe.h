#ifdef __cplusplus
extern "C" {
#endif

//Non-rentrant rdbe connection
void rdbe_connect(void);
void rdbe_disconnect(void);
void rdbe_free(void *);

/**Gather a set of Packets
 * Memory dynamically allocated
 * Memory must be deallocated by caller
 * using rdbe_free
 * Will be length (VDIFF_SIZE-sizeof(vheader_t))*Packets
 */
const int8_t *rdbe_gather(size_t Packets);

#ifdef __cplusplus
}
#endif
