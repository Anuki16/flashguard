import cv2
import time
import numpy as np
from hazard_detect import saturated_red_flash_count, luminance_flash_count
from collections import deque
import imutils
from multiprocessing import Process, Lock
import sys

# Buffer size (in frames)
BUFFER_SIZE = 16

# Disable auto exposure
# TODO: This is not working

if __name__ == '__main__':
    # Open the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Width and height of the webcam frame after resizing (without changing resolution)
    ret, frame = cap.read()
    print('here')
    frame = imutils.resize(frame, width = frame.shape[1] // 4)
    
    WIDTH = frame.shape[1]
    HEIGHT = frame.shape[0]

    # Set the quarter area threshold (in pixels)
    quarter_area_threshold = (WIDTH * HEIGHT) / 4
    
    # Initialize the frame buffer
    frame_buffer = deque()
    time_buffer = deque()
    luminous_flash_buffer = deque()
    red_flash_buffer = deque()
    
    luminous_flashes = np.zeros(frame.shape[:2])
    red_flashes = np.zeros(frame.shape[:2])
    
    # Initialize previous time
    prev_time = time.time()

# mutexes
cam_mutex = Lock()
frame_buf_mutex = Lock()
flash_buf_mutex = Lock()

def thread_action():
    global luminous_flashes, red_flashes
    while True:
        # Read a frame from the webcam
        with cam_mutex:
            ret, frame = cap.read()
        
        #If the frame was successfully read
        if ret:
            s = time.time()
            #changing the resolution while keeping the aspect ratio
            cur_frame = imutils.resize(frame, width=WIDTH)
            cur_time = time.time()
            
            # Add the frame to the frame buffer
            with frame_buf_mutex:
                if frame_buffer:
                    prev_frame = frame_buffer[-1]
                else:
                    prev_frame = None
                frame_buffer.append(cur_frame)
                time_buffer.append(cur_time)
            
            # If a previous frame exists, check whether the transition has a luminous/red flash
            if prev_frame is not None:
                # Check if transition is a flash
                luminous = luminance_flash_count(cur_frame, prev_frame)
                red = saturated_red_flash_count(cur_frame, prev_frame)
                
                # Update buffers and flash count
                with flash_buf_mutex:
                    luminous_flash_buffer.append(luminous)
                    luminous_flashes += luminous
                    red_flash_buffer.append(red)
                    red_flashes += red
                
            # When there are enough frames and more than 1s has elapsed, check for flashing sequences
            interval = cur_time - time_buffer[0]
            if len(frame_buffer) >= BUFFER_SIZE and interval >= 1:
                
                luminous_flash_freq = (luminous_flashes/2) / interval
                red_flash_freq = (red_flashes/2) / interval
                
                luminous_count = np.sum(luminous_flash_freq >= 3)
                red_count = np.sum(red_flash_freq >= 3)
                
                print(np.mean(luminous_flash_freq), np.mean(red_flash_freq))
                if luminous_count >= quarter_area_threshold or red_count >= quarter_area_threshold:
                    print("Flashing detected")
                sys.stdout.flush()
                # Pop first element to change sliding window
                with frame_buf_mutex:
                    frame_buffer.popleft()
                    time_buffer.popleft()
                
                # Update flash count
                with flash_buf_mutex:
                    luminous_flashes -= luminous_flash_buffer.popleft()
                    red_flashes -= red_flash_buffer.popleft()
    
if __name__ == '__main__':
    try:
        threads = [0] * 4
        for i in range(4):
            threads[i] = Process(target = thread_action)
            threads[i].start()
            time.sleep(1/60)
            
        for i in range(4):
            threads[i].join()
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
