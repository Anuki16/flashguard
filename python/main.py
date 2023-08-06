import cv2
import time
import numpy as np
from hazard_detect import is_hazardous
from collections import deque

# Buffer size (in frames)
BUFFER_SIZE = 16

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 100)

# Disable auto exposure
# TODO: This is not working

# Reduce camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH//4)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT//4)

# Width and height of the webcam frame
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the quarter area threshold (in pixels)
quarter_area_threshold = (WIDTH * HEIGHT) / 4

# Initialize the frame buffer
frame_buffer = []
time_buffer = []


# Initialize previous time
prev_time = time.time()


# Capture frames from the webcam
try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        #If the frame was successfully read
        if ret:
            
            # Add the frame to the frame buffer
            frame_buffer.append(frame)
            time_buffer.append(time.time())
    
            # If the frame buffer has at least 8 frames
            if len(frame_buffer) >= BUFFER_SIZE:
    
                # Get the current FrameSlidingWindow
                frame_sliding_window = frame_buffer[-BUFFER_SIZE:]
                flash_count = 0
                # For each pixel location in frame
                for j in range(1, BUFFER_SIZE):
                    # Initialize the flashing pixels count
                    flashing_pixels_count = 0
                    # Get the current and previous frames
                    cur_frame = frame_sliding_window[j]
                    prev_frame = frame_sliding_window[j-1]
    
                    flashing_pixels_count = np.sum(is_hazardous(cur_frame / 255, prev_frame / 255))
                    if (flashing_pixels_count > quarter_area_threshold):
                        flash_count += 1
                flash_freq = (flash_count/2)/(time_buffer[-1] - time_buffer[-BUFFER_SIZE])
                print(flash_freq)
                if flash_freq >= 3:
                    print("Flashing detected")
            
finally:
    cap.release()
    cv2.destroyAllWindows()
    
