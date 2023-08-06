
import cv2
import time
import numpy as np
from hazard_detect import is_hazardous
from collections import deque
import imutils
import serial
import serial.tools.list_ports

# CONSTANTS
PORT = '/dev/rfcomm0'
# Buffer size (in frames)
BUFFER_SIZE = 8

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

def run_program():
    serial_connection = initialise_bluetooth_communication();

    # Open the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 100)

    # Disable auto exposure
    # TODO: This is not working

    # # Reduce camera resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    # Width and height of the webcam frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=160)

    WIDTH = 160
    HEIGHT = frame.shape[0]

    # WIDTH = 160 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # HEIGHT = 90 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            if (serial_connection == None):
                serial_connection = initialise_bluetooth_communication();

            #check if there is incoming data
            while serial_connection != None and check_for_connection() and serial_connection.in_waiting:
                receive_data(serial_connection)

            # Read a frame from the webcam
            ret, frame = cap.read()
            
            #changing the resolution while keeping the aspect ratio
            frame = imutils.resize(frame, width=160)

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
        
                        # For each pixel in frame
                        # for k in range(frame.shape[0]):  # TODO: This is too slow
                        #     for l in range(frame.shape[1]):
                        #         if is_hazardous(cur_frame[k][l] / 255, prev_frame[k][l] / 255):
                        #             flashing_pixels_count += 1
                        flashing_pixels_count = np.sum(is_hazardous(cur_frame / 255, prev_frame / 255))
                        if (flashing_pixels_count > quarter_area_threshold):
                            flash_count += 1
                    flash_freq = (flash_count/2)/(time_buffer[-1] - time_buffer[-BUFFER_SIZE])
                    print(flash_freq)
                    send_array_over_bluetooth([round(flash_freq,2)], serial_connection)

                    print("a",(time_buffer[-1] - time_buffer[0])/(len(time_buffer)-1))
                    if flash_freq >= 3:
                        print("Flashing detected")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


while True:
    try:
        run_program()
    except Exception as e:
        print(e)
