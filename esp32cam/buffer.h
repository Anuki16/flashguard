
const int	WIDTH = 96;
const int	HEIGHT = 96;
const int BUFFER_SIZE = 16;

typedef struct {
  uint8_t buffer[BUFFER_SIZE][HEIGHT][WIDTH];
  uint8_t rp = 0;
  uint8_t wp = 0;
  uint8_t size = 0;
} frame_buf;

typedef struct {
  unsigned long buffer[BUFFER_SIZE];
  uint8_t rp = 0;
  uint8_t wp = 0;
  uint8_t size = 0;
} time_buf;


void frame_push(frame_buf* buf) {
  // Copy the data to the buffer at the write pointer position
  // Update the write pointer and size
  buf->wp = (buf->wp + 1) % BUFFER_SIZE;
  buf->size = (buf->size < BUFFER_SIZE) ? buf->size + 1 : BUFFER_SIZE;
}

void frame_pop(frame_buf* buf) {
  // Update the read pointer and size
  buf->rp = (buf->rp + 1) % BUFFER_SIZE;
  buf->size = (buf->size > 0) ? buf->size - 1 : 0;
}

void frame_init(frame_buf* buf) {
  buf->rp = 0;
  buf->wp = 0;
  buf->size = 0;
}

void time_init(time_buf* buf) {
  buf->rp = 0;
  buf->wp = 0;
  buf->size = 0;
}

void time_push(time_buf* buf, unsigned long timestamp) {
  // Copy the data to the buffer at the write pointer position
  buf->buffer[buf->wp] = timestamp;

  // Update the write pointer and size
  buf->wp = (buf->wp + 1) % BUFFER_SIZE;
  buf->size = (buf->size < BUFFER_SIZE) ? buf->size + 1 : BUFFER_SIZE;
}

void time_pop(time_buf* buf) {
  // Update the read pointer and size
  buf->rp = (buf->rp + 1) % BUFFER_SIZE;
  buf->size = (buf->size > 0) ? buf->size - 1 : 0;
}

unsigned long time_front(time_buf* buf) {
  return buf->buffer[buf->rp];
}
