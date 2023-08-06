#include "esp_camera.h"
#include <WiFi.h>

//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//
//            You must select partition scheme from the board menu that has at least 3MB APP space.
//            Face Recognition is DISABLED for ESP32 and ESP32-S2, because it takes up from 15 
//            seconds to process single frame. Face Detection is ENABLED if PSRAM is enabled as well

// ===================
// Select camera model
// ===================
#define CAMERA_MODEL_AI_THINKER // Has PSRAM

#include "camera_pins.h"
#include "esp_camera.h"
#include "buffer.h"
#include <cstdlib>
#include <cstring>
#include "BluetoothSerial.h"

#define R 0
#define G 1
#define B 2
#define FRAME_SIZE FRAMESIZE_96X96
#define GLASS_PIN 15
#define LED 4

using namespace std;

// function prototyping
void flash_counts(float color[], float prev_color[], float flash[]);
void sRGB_to_linearRGB(float sRGB[]);
float inverse_gamma_transform(float signal);
float linearRBG_to_Ls(float linearRGB[]);
int is_luminance_flash(float ls, float prev_ls);
float red_ratio(float sRGB[]);
float pure_red(float sRGB[]);
void camera_setup();

BluetoothSerial SerialBT;

camera_fb_t * fb = NULL;  // pointer to image taken by camera

uint8_t* frame;
uint8_t* prev_frame;

time_buf* time_buffer;
unsigned long prev_time = 0;
int fcount = 0;

frame_buf* luminous_flash_buffer;
frame_buf* red_flash_buffer;

uint8_t* luminous_flashes;
uint8_t* red_flashes;

uint8_t* flash;

const int	quarter_area_threshold = (WIDTH * HEIGHT) / 6;

unsigned long s;

uint8_t glasses_on = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  camera_setup();
  psramInit();
  pinMode(GLASS_PIN, OUTPUT);
  SerialBT.begin("FLASHGUARD");

  frame = (uint8_t*) ps_malloc(WIDTH * HEIGHT * 3 * sizeof(uint8_t));
  prev_frame = (uint8_t*) ps_malloc(WIDTH * HEIGHT * 3 * sizeof(uint8_t));
  time_buffer = (time_buf*) ps_malloc(sizeof(time_buf));
  time_init(time_buffer);
  
  luminous_flash_buffer = (frame_buf*) ps_malloc(sizeof(frame_buf));
  frame_init(luminous_flash_buffer);
  red_flash_buffer = (frame_buf*) ps_malloc(sizeof(frame_buf));
  frame_init(red_flash_buffer);

  luminous_flashes = (uint8_t*) ps_malloc(WIDTH * HEIGHT * sizeof(uint8_t));
  memset(luminous_flashes, 0, WIDTH * HEIGHT);
  red_flashes = (uint8_t*) ps_malloc(WIDTH * HEIGHT * sizeof(uint8_t));
  memset(red_flashes, 0, WIDTH * HEIGHT);
  flash = new uint8_t[2];

}

void loop() {
  // put your main code here, to run repeatedly:
  
  if (millis() - prev_time >= 1000) {
    Serial.println(fcount);
    fcount = 0;
    prev_time = millis();
  }

  fb = esp_camera_fb_get();  
  if(!fb) {
    Serial.println("Camera capture failed");
  } else {
    // copy data into frame 
    save_frame();
    time_push(time_buffer, millis());
    //Serial.printf("Buffer: %d %d\n", luminous_flash_buffer->rp, luminous_flash_buffer->wp);
    if (time_buffer->size > 1) {
      s = millis();
      calc_flashes();
      //Serial.printf("Time to check if flashing: %d\n",millis() - s);
      unsigned long interval = millis() - time_front(time_buffer);
      if (time_buffer->size == BUFFER_SIZE || interval >= 1000) {
        int luminous_count = 0;
        int red_count = 0;
        float luminous_freq;
        float red_freq;
        float lum_sum = 0;
        float red_sum = 0;
        s = millis();
        for (int i = 0; i < HEIGHT; i++) {
          for (int j = 0; j < WIDTH; j++) {
            luminous_freq = ((float)luminous_flashes[i*WIDTH + j]/2.0)/(float(interval)/1000.0);
            //red_freq = ((float)red_flashes[i*WIDTH + j]/2.0)/(float(interval)/1000.0);
            if (luminous_freq >= 2) luminous_count++;
            //if (red_freq >= 3) red_count++;
            lum_sum += luminous_freq;
            //red_sum += red_freq;

            luminous_flashes[i*WIDTH + j] -= luminous_flash_buffer->buffer[luminous_flash_buffer->rp][i][j];
            //red_flashes[i*WIDTH + j] -= red_flash_buffer->buffer[red_flash_buffer->rp][i][j];
            if (i == 50 && j == 100) {
              //Serial.printf("%f %f %d %d RP: %d\n", luminous_freq, red_freq, luminous_flashes[i*WIDTH + j], red_flashes[i*WIDTH + j], luminous_flash_buffer->rp);
            }
          }
        }
        Serial.printf("%d %d %f %f\n",luminous_count, red_count, lum_sum/(float)(WIDTH*HEIGHT), red_sum/(float)(WIDTH*HEIGHT));
        glasses_on = (glasses_on << 1);
        if (luminous_count >= quarter_area_threshold ){//|| red_count >= quarter_area_threshold){
          SerialBT.println(lum_sum/(float)(WIDTH*HEIGHT));
          glasses_on |= 1;
        }
        if (glasses_on) {
          digitalWrite(GLASS_PIN, LOW);
          delay(5);
          digitalWrite(GLASS_PIN, HIGH);
        } else {
          digitalWrite(GLASS_PIN, LOW);
        }
        
        time_pop(time_buffer);
        frame_pop(luminous_flash_buffer);
        //frame_pop(red_flash_buffer);
        //Serial.printf("Time to check complete: %d\n",millis() - s);
      }
    }
    memcpy(prev_frame, frame, WIDTH * HEIGHT * 3 * sizeof(uint8_t));
  }
  esp_camera_fb_return(fb); 
  fcount++;
}

void calc_flashes() {
  //uint8_t* color = frame + 4*WIDTH*3 + 10*3;
  //Serial.printf("%d %d %d -",color[0], color[1], color[2]);
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
      flash_counts(frame + i*WIDTH*3 + j*3, prev_frame + i*WIDTH*3 + j*3, flash);
      //Serial.print(flash[0]);
      luminous_flash_buffer->buffer[luminous_flash_buffer->wp][i][j] = flash[0];
      luminous_flashes[i*WIDTH + j] += flash[0];
      //red_flash_buffer->buffer[red_flash_buffer->wp][i][j] = flash[1];
      //red_flashes[i*WIDTH + j] += flash[1];
      
      if (i == 50 && j == 100) {
        //Serial.println(linearRBG_to_Ls(frame + i*WIDTH*3 + j*3));
        //Serial.println(linearRBG_to_Ls(prev_frame + i*WIDTH*3 + j*3));
        //Serial.printf("%d %d , %d %d\n", flash[0], flash[1], luminous_flashes[i*WIDTH + j], red_flashes[i*WIDTH + j]);
      }
    }
    //Serial.println();
  }
  frame_push(luminous_flash_buffer);
  //frame_push(red_flash_buffer);
}

void save_frame() {
  uint8_t r, g, b, row, col;
  for (int i = 0; i < fb->len; i+=2) {
      uint16_t rgb565 = (uint16_t)(fb->buf)[i] << 8 | (uint16_t)(fb->buf)[i+1];
      
      r = (rgb565 >> 11) & 0x1F;
      g = (rgb565 >> 5) & 0x3F;
      b = rgb565 & 0x1F;
    
      row = (i/2)/WIDTH;
      col = (i/2)%WIDTH;
      frame[row*WIDTH*3 + col*3 + R] = (r << 3) | (r >> 2);
      frame[row*WIDTH*3 + col*3 + G] = (g << 2) | (g >> 4);
      frame[row*WIDTH*3 + col*3 + B] = (b << 3) | (b >> 2);
      
  }
}

void camera_setup() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if(config.pixel_format == PIXFORMAT_JPEG){
    if(psramFound()){
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // Special settings
	config.frame_size = FRAME_SIZE;
  config.pixel_format = PIXFORMAT_RGB565;

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1); // flip it back
    s->set_brightness(s, 1); // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
  }
}



float linearRBG_to_Ls(uint8_t linearRGB[]) {
	return (0.2126 * (float)linearRGB[R] + 0.7152 * (float)linearRGB[G] + 0.0722 * (float)linearRGB[B]) / 255;
}

int is_luminance_flash(float ls, float prev_ls) {
	float brighter_ls = max(ls, prev_ls);
  float darker_ls = min(ls, prev_ls);
  
  if (brighter_ls - darker_ls >= 0.1 && darker_ls < 0.8) return 1;
  return 0;
}

int is_saturated_red_flash(uint8_t linear_color[], uint8_t prev_linear_color[]) {
	if (red_ratio(linear_color) >= 0.8 || red_ratio(prev_linear_color) >= 0.8) {
    if (abs(pure_red(linear_color) - pure_red(prev_linear_color)) >= 20) return 1;
  }
  return 0;
}

void flash_counts(uint8_t color[], uint8_t prev_color[], uint8_t* flash) {

  float ls = linearRBG_to_Ls(color);
  float prev_ls = linearRBG_to_Ls(prev_color);

  flash[0] = is_luminance_flash(ls, prev_ls);
  flash[1] = 0;//is_saturated_red_flash(color, prev_color);
}

float red_ratio(uint8_t sRGB[]) {
	float result = sRGB[R]/(sRGB[R] + sRGB[G] + sRGB[B] + 1e-10);
	return result;
}

float pure_red(uint8_t sRGB[]) {
	float comp = (float) (sRGB[R] - sRGB[G] - sRGB[B]) / 255.0;
  return max(comp * 320, (float) 0);
}