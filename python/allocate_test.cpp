#include <cstdint>
#include <cstring>
#include <iostream>
#include <queue>


using namespace std;

uint8_t* frame = NULL;

const int BUFFER_SIZE = 16;
const float frame_scaling_factor = 0.25;
const int	frame_width = 352;
const int	frame_height = 288;

const int	quarter_area_threshold = (frame_width * frame_height) / 4;

// buffers 
deque<uint8_t*> frame_buffer;
deque<time_t> time_buffer;
deque<float[frame_width][frame_height]> luminous_flash_buffer;
deque<float[frame_width][frame_height]> red_flash_buffer;

float luminous_flashes[frame_height][frame_width] = {0};
float red_flashes[frame_height][frame_width] = {0};


uint8_t*** allocate_3d(int width, int length, int height) {
    cout << width << length << height << endl;
  uint8_t*** frame = new uint8_t**[width]();
  if (!frame) cout << "Allocation failed" <<endl;

  for (int i = 0; i < width; i++) {
    frame[i] = new uint8_t*[length]();
    if (!frame[i]) cout << "Allocation failed" <<endl;

    for (int j = 0; j < length; j++) {
      frame[i][j] = new uint8_t[height]();
      if (!frame[i][j]) cout << "Allocation failed" <<endl;
    }
  }
  
  return frame;
}

void deallocate_3d(int width, int length, int height, uint8_t*** frame) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < length; j++) {
      delete[] frame[i][j];
    }
    delete[] frame[i];
  }
  delete[] frame;
}

uint8_t access(uint8_t* frame, int i, int j, int k) {
    return frame[i * 300 * 3 + j * 3 + k];
}

int main() {
    uint8_t* buf = new uint8_t[200*300*3];
    for (int i = 0; i < 200*300*3; i++) {
        buf[i] = (i+1)%256;
    }

    uint8_t* frame = new uint8_t[200*300*3];


    memcpy(frame, buf, 200*300);//200*300*3*sizeof(uint8_t));

    cout << "here" << endl;
    cout << (int)access(frame, 0, 0,0) << endl;
    cout << "here" << endl;

    delete[] frame;
    delete[] buf;

    return 0;
}