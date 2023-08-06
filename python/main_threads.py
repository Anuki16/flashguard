# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 23:04:22 2023

@author: anuki
"""

import cv2
import time
import numpy as np
from hazard_detect import saturated_red_flash_count, luminance_flash_count
from collections import deque
import imutils
import threading

# Buffer size (in frames)
BUFFER_SIZE = 16

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Disable auto exposure
# TODO: This is not working

# Width and height of the webcam frame after resizing (without changing resolution)
ret, frame = cap.read()
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


def capture_frames():
    while True:
        
        if len(time_buffer) > 60:
            #print(time_buffer[-1] - time_buffer[0])
            continue
        ret, frame = cap.read()
        #If the frame was successfully read
        if ret:
            #changing the resolution while keeping the aspect ratio
            frame = imutils.resize(frame, width=WIDTH)
            # Add the frame to the frame buffer
            frame_buffer.append(frame)
            time_buffer.append(time.time())
        
        
def process_frames():
    global luminous_flashes, red_flashes
    while True:
        # If a previous frame exists, check whether the transition has a luminous/red flash
        s = time.time()
        if len(frame_buffer) > 1:
            cur_frame = frame_buffer[-1]
            prev_frame = frame_buffer[-2]
            
            # Check if transition is a flash
            luminous = luminance_flash_count(cur_frame, prev_frame)
            red = saturated_red_flash_count(cur_frame, prev_frame)
            
            # Update buffers and flash count
            luminous_flash_buffer.append(luminous)
            luminous_flashes += luminous
            red_flash_buffer.append(red)
            red_flashes += red
            print(time.time() - s)
            
        # When there are enough frames and more than 1s has elapsed, check for flashing sequences
        interval = time_buffer[-1] - time_buffer[0] if len(time_buffer) > 0 else 0
        if len(frame_buffer) >= BUFFER_SIZE and interval >= 1:
            
            luminous_flash_freq = (luminous_flashes/2) / interval
            red_flash_freq = (red_flashes/2) / interval
            
            luminous_count = np.sum(luminous_flash_freq >= 3)
            red_count = np.sum(red_flash_freq >= 3)
            
            print(len(time_buffer), interval, np.mean(luminous_flash_freq), np.mean(red_flash_freq))
            if luminous_count >= quarter_area_threshold or red_count >= quarter_area_threshold:
                print("Flashing detected")
                
            # Pop first element to change sliding window
            frame_buffer.popleft()
            time_buffer.popleft()
            
            # Update flash count
            luminous_flashes -= luminous_flash_buffer.popleft()
            red_flashes -= red_flash_buffer.popleft()


try:
    # creating threads
    t1 = threading.Thread(target=process_frames, name='t1')
    t2 = threading.Thread(target=capture_frames, name='t2') 
 
    # starting threads
    t1.start()
    t2.start()
 
    # wait until all threads finish
    t1.join()
    t2.join()
finally:
    cap.release()
    cv2.destroyAllWindows()
    
