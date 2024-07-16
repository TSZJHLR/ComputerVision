# import the necessary modules
import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# function to get the current audio endpoint volume interface
def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# function to set the volume level
def set_volume(level):
    volume = get_volume_interface()
    volume.SetMasterVolumeLevelScalar(level, None)

# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# initialize opencv video capture
cap = cv2.VideoCapture(0)

# ground truth and predictions for metric evaluation
ground_truth = []
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # preprocessing: apply gaussian blur
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)

    # segmentation: thresholding to create a binary mask
    _, mask = cv2.threshold(frame_gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)

        # post-processing: draw the largest contour
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # check if hand landmarks are detected
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip_x = int(thumb_tip.x * frame.shape[1])
                thumb_tip_y = int(thumb_tip.y * frame.shape[0])
                index_tip_x = int(index_tip.x * frame.shape[1])
                index_tip_y = int(index_tip.y * frame.shape[0])
                distance = math.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)
                volume_level = np.interp(distance, [30, 300], [0.0, 1.0])
                set_volume(volume_level)
                cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)
                cv2.line(frame, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 255, 0), 2)

                # classification: recognize pinch gesture
                pinch_gesture = distance < 50
                predictions.append(pinch_gesture)
                ground_truth.append(True)  # assuming we expect a pinch gesture for this demo

    # display the frame
    cv2.imshow('Volume Control', frame)

    # break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and close all opencv windows
cap.release()
cv2.destroyAllWindows()

# metric evaluation
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f'accuracy: {accuracy:.2f}')
print(f'precision: {precision:.2f}')
print(f'recall: {recall:.2f}')
print(f'f1-score: {f1:.2f}')
