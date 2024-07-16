import cv2
import mediapipe as mp
import numpy as np
import math

# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# initialize opencv video capture
cap = cv2.VideoCapture(0)

# set up drawing parameters
drawing = False
brush_size = 5
eraser_size = 50
last_x, last_y = None, None
canvas = None

def draw_circle(frame, x, y, size, color):
    cv2.circle(frame, (x, y), size, color, -1)

def draw_line(frame, x1, y1, x2, y2, size, color):
    cv2.line(frame, (x1, y1), (x2, y2), color, int(size))  # cast size to integer

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_x = int(thumb_tip.x * frame.shape[1])
    thumb_tip_y = int(thumb_tip.y * frame.shape[0])
    index_tip_x = int(index_tip.x * frame.shape[1])
    index_tip_y = int(index_tip.y * frame.shape[0])
    distance = math.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)

    # detect pinch gesture for drawing
    if distance < 40:
        return 'pinch', index_tip_x, index_tip_y

    # detect fist gesture for erasing
    if all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y for i in range(9, 21)):
        return 'fist', None, None

    return None, None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if canvas is None:
        canvas = np.zeros_like(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture, x, y = detect_gesture(hand_landmarks)

            if gesture == 'pinch':
                drawing = True
                if last_x is None or last_y is None:
                    last_x, last_y = x, y
                draw_line(canvas, last_x, last_y, x, y, brush_size, (255, 0, 0))
                last_x, last_y = x, y

            elif gesture == 'fist':
                drawing = False
                canvas = np.zeros_like(frame)  # clear the canvas when a fist is detected
                last_x, last_y = None, None  # reset last coordinates

            else:
                last_x, last_y = None, None

    frame = cv2.add(frame, canvas)
    cv2.imshow('Art Canvas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
