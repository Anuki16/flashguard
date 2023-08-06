import cv2
import time
import numpy as np
from hazard_detect import saturated_red_flash_count, luminance_flash_count
from collections import deque
import imutils
import serial
import serial.tools.list_ports
from multiprocessing import Pool
import RPi.GPIO as GPIO

# CONSTANTS
PORT = '/dev/rfcomm0'
BUFFER_SIZE = 16 # Buffer size (in frames)

def initialise_bluetooth_communication():
    #initialising the serial communication 
    if(check_for_connection()):
        serial_connection = serial.Serial(
            port=PORT,
            timeout=0.01,
            writeTimeout=1,
        );
        
        return serial_connection

def send_array_over_bluetooth(array, serial_connection):
    if(check_for_connection() and serial_connection != None):
        array_string = list(map(str, array))
        data_string = ",".join(array_string) + "\n"
        data_bytes = data_string.encode('utf-8')

        serial_connection.write(data_bytes)
        serial_connection.flush()
    else:
        print("WARNING: No Bluetooth Connection")
    
def receive_data(serial_connection):
    print("Data received")
    print(serial_connection.readline())

def check_for_connection():
    available_ports = [tuple(port)[0] for port in list(serial.tools.list_ports.comports())]

    if PORT in available_ports:
        return True

    return False

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

def run_program():
    serial_connection = initialise_bluetooth_communication();

    # Capture frames from the webcam
    try:
        while True:
            if (serial_connection == None):
                serial_connection = initialise_bluetooth_communication();

            #check if there is incoming data
            while serial_connection != None and check_for_connection() and serial_connection.in_waiting:
                receive_data(serial_connection)
            
            while not time_buffer or time_buffer[-1] - time_buffer[0] < 0.6:#len(time_buffer) < 30:
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
                send_array_over_bluetooth(["Flashing detected","Luminance flashing",np.mean(flash_freqs[0]), "Red flashing", np.mean(flash_freqs[1])], serial_connection)
                GPIO.output(14,GPIO.HIGH)
            else:
                GPIO.output(14,GPIO.LOW)
                
            # Pop first element to change sliding window
            frame_buffer.clear()
            time_buffer.clear()
                
                
    finally:
        print(time_buffer) 
        cap.release()
        cv2.destroyAllWindows()


# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)

# Setup rpi LED
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(14,GPIO.OUT)

# Disable auto exposure
# TODO: This is not working

# Width and height of the webcam frame after resizing (without changing resolution)
ret, frame = cap.read()
frame = imutils.resize(frame, width = frame.shape[1] // 5)

WIDTH = frame.shape[1]
HEIGHT = frame.shape[0]

# Set the quarter area threshold (in pixels)
quarter_area_threshold = (WIDTH * HEIGHT) / 4

# Initialize the frame buffer
frame_buffer = []
time_buffer = []
run_program()
