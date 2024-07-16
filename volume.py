import cv2  # import the OpenCV library for computer vision tasks
import mediapipe as mp  # import Mediapipe for hand tracking
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # import Pycaw for audio control
from ctypes import cast, POINTER  # import necessary modules for ctypes casting
from comtypes import CLSCTX_ALL  # import CLSCTX_ALL for comtypes
import numpy as np  # import numpy for numerical operations
import math  # import math module for mathematical operations
import time  # import time module for time-related operations

# function to get the current audio endpoint volume interface
def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()  # get the audio devices (speakers)
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # activate the audio endpoint volume interface
    volume = cast(interface, POINTER(IAudioEndpointVolume))  # cast the interface to IAudioEndpointVolume pointer
    return volume  # return the volume interface

# function to set the volume level
def set_volume(level):
    volume = get_volume_interface()  # get the current volume interface
    volume.SetMasterVolumeLevelScalar(level, None)  # set the master volume level scalar

# initialize Mediapipe hands
mp_hands = mp.solutions.hands  # initialize Mediapipe Hands
hands = mp_hands.Hands(max_num_hands=1)  # set maximum number of hands to detect
mp_draw = mp.solutions.drawing_utils  # import Mediapipe drawing utilities

# initialize OpenCV video capture using the default camera (0)
cap = cv2.VideoCapture(0)

# variables for FPS calculation and volume control
prev_time = 0  # initialize previous time for FPS calculation
fps = 0  # initialize FPS
volume_level = 0.5  # initial volume level
min_distance = 50  # minimum distance in pixels from camera
max_distance = 300  # maximum distance in pixels from camera

while cap.isOpened():  # loop while video capture is open
    ret, frame = cap.read()  # read a frame from the video capture
    if not ret:  # if frame reading fails, break loop
        break

    frame = cv2.flip(frame, 1)  # flip frame horizontally for natural viewing

    current_time = time.time()  # get current time for FPS calculation
    fps = 1 / (current_time - prev_time)  # calculate FPS
    prev_time = current_time  # update previous time for FPS calculation

    # convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)  # process the frame with Mediapipe Hands
    if result.multi_hand_landmarks:  # if hand landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:
            # draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # calculate thumb and index finger positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            thumb_tip_y = int(thumb_tip.y * frame.shape[0])
            index_tip_x = int(index_tip.x * frame.shape[1])
            index_tip_y = int(index_tip.y * frame.shape[0])
            
            # calculate distance between thumb and index finger
            distance = math.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)
            
            # interpolate volume level based on distance within defined range
            if min_distance <= distance <= max_distance:
                volume_level = np.interp(distance, [min_distance, max_distance], [0.0, 1.0])  # interpolate volume level
                set_volume(volume_level)  # set the volume level
            
            # draw circles and line between thumb and index finger on the frame
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 0), -1)  # draw circle for thumb tip
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)  # draw circle for index finger tip
            cv2.line(frame, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 255, 0), 2)  # draw line between thumb and index finger

    # display volume percentage in top-left corner of the frame
    cv2.putText(frame, f'Volume: {int(volume_level * 100)}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # draw vertical volume bar on the left side of the frame
    bar_height = int(volume_level * (frame.shape[0] - 100))  # calculate bar height based on volume level
    cv2.rectangle(frame, (10, frame.shape[0] - 100), (30, frame.shape[0] - 100 - bar_height), (0, 255, 0), -1)  # draw rectangle for volume bar

    # display FPS in top-right corner of the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # display the frame with annotations
    cv2.imshow('Volume Control', frame)

    # break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
