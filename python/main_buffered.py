import cv2
import time
import numpy as np
from hazard_detect import saturated_red_flash_count, luminance_flash_count
from collections import deque
import imutils
from multiprocessing import Pool

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
frame_buffer = []
time_buffer = []


def thread_process_frames(proc_range):
    luminous_flashes = np.zeros(frame.shape[:2])
    red_flashes = np.zeros(frame.shape[:2])
    #print(a, b, "Starting", flush = True)
    for i in range(proc_range[0] + 1, min(proc_range[1], len(frame_buffer))):
        # If a previous frame exists, check whether the transition has a luminous/red flash
        
        cur_frame = frame_buffer[i]
        prev_frame = frame_buffer[i - 1]
        
        # Check if transition is a flash
        luminous = luminance_flash_count(cur_frame, prev_frame)
        red = saturated_red_flash_count(cur_frame, prev_frame)
        
        # Update buffers and flash count
        
        luminous_flashes += luminous
        red_flashes += red
    #print(a, b, "Done", flush = True)
    return np.array([luminous_flashes, red_flashes])

# Capture frames from the webcam
try:
    while True:
        
        while not time_buffer or time_buffer[-1] - time_buffer[0] < 1:#len(time_buffer) < 30:
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            #If the frame was successfully read
            if ret:
                #changing the resolution while keeping the aspect ratio
                frame = imutils.resize(frame, width=WIDTH)
                # Add the frame to the frame buffer
                frame_buffer.append(frame)
                time_buffer.append(time.time())
        s = time.time()
        
        n = len(frame_buffer)
        per_thread = int(np.ceil(n/4))
        ranges = [(i, i+per_thread) for i in range(0, n, per_thread)]
        
        with Pool(4) as pool:
            flashes = np.sum(pool.map(thread_process_frames, ranges), axis = 0)
        
        print("Time for processing %d frames = %f"%(n, time.time() - s))
        
        # When there are enough frames and more than 1s has elapsed, check for flashing sequences
        interval = time_buffer[-1] - time_buffer[0]
        flash_freqs = (flashes/2) / interval
        
        luminous_count = np.sum(flash_freqs[0] >= 3)
        red_count = np.sum(flash_freqs[1] >= 3)
        
        print(np.mean(flash_freqs[0]), np.mean(flash_freqs[1]))
        if luminous_count >= quarter_area_threshold or red_count >= quarter_area_threshold:
            print("Flashing detected, no of flashing pixels = ", luminous_count, red_count)
        
        # Pop first element to change sliding window
        frame_buffer.clear()
        time_buffer.clear()
        
        
finally:
    cap.release()
    cv2.destroyAllWindows()
    
