import cv2  # import opencv library for image processing
import mediapipe as mp  # import mediapipe for hand tracking
import numpy as np  # import numpy for numerical operations
import math  # import math module for mathematical operations

# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # initialize hands tracker with max 1 hand
mp_draw = mp.solutions.drawing_utils  # util functions for drawing landmarks

# initialize opencv video capture
cap = cv2.VideoCapture(0)  # capture video from default camera (index 0)

# set up drawing parameters
drawing = False  # flag to indicate if drawing
brush_size = 5  # initial size of the brush
eraser_size = 50  # size of the eraser
last_x, last_y = None, None  # last drawn position
canvas = None  # canvas to draw on
color = (255, 0, 0)  # initial brush color (in BGR format)

# function to draw a filled circle on a frame
def draw_circle(frame, x, y, size, color):
    cv2.circle(frame, (x, y), size, color, -1)

# function to draw a line on a frame
def draw_line(frame, x1, y1, x2, y2, size, color):
    cv2.line(frame, (x1, y1), (x2, y2), color, int(size))  # draw line with specified size

# function to detect gestures based on hand landmarks
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]  # thumb tip landmark
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # index finger tip landmark
    thumb_tip_x = int(thumb_tip.x * frame.shape[1])  # x coordinate of thumb tip
    thumb_tip_y = int(thumb_tip.y * frame.shape[0])  # y coordinate of thumb tip
    index_tip_x = int(index_tip.x * frame.shape[1])  # x coordinate of index finger tip
    index_tip_y = int(index_tip.y * frame.shape[0])  # y coordinate of index finger tip
    distance = math.sqrt((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2)  # distance between thumb and index finger tips

    # detect pinch gesture for drawing
    if distance < 40:
        return 'pinch', index_tip_x, index_tip_y

    # detect fist gesture for erasing
    if all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y for i in range(9, 21)):
        return 'fist', None, None

     # detect open hand gesture for color change
    if all(hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in range(9, 21)):
        return 'open', None, None

    return None, None, None

# function to change brush color
def change_color():
    global color
    # rotate through predefined colors
    if color == (255, 0, 0):
        color = (0, 255, 0)
    elif color == (0, 255, 0):
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

# main loop for capturing video and processing gestures
while cap.isOpened():
    ret, frame = cap.read()  # read a frame from the camera
    if not ret:
        print("failed to capture frame. exiting...")
        break

    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB (Mediapipe uses RGB)

    # process the frame with Mediapipe Hands
    result = hands.process(frame_rgb)

    if canvas is None:
        canvas = np.zeros_like(frame)  # initialize canvas if not already

    # if hands are detected in the frame, process each detected hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # draw landmarks and connections

            # detect gestures and perform corresponding actions
            gesture, x, y = detect_gesture(hand_landmarks)

            if gesture == 'pinch':
                drawing = True
                if last_x is None or last_y is None:
                    last_x, last_y = x, y
                draw_line(canvas, last_x, last_y, x, y, brush_size, color)
                last_x, last_y = x, y

            elif gesture == 'fist':
                drawing = False
                canvas = np.zeros_like(frame)  # clear the canvas when a fist is detected
                last_x, last_y = None, None  # reset last coordinates

            elif gesture == 'open':
                change_color()  # change brush color
                last_x, last_y = None, None  # reset last coordinates

            else:
                last_x, last_y = None, None  # reset last coordinates

    # add the canvas (drawing) on top of the frame
    frame = cv2.add(frame, canvas)

    # display the frame with the drawing on a window named "Art Canvas"
    cv2.imshow('Art Canvas', frame)

    # wait for 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # release the video capture object
cv2.destroyAllWindows()  # close all OpenCV windows
