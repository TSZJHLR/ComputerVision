import cv2  # import opencv library for image processing
import mediapipe as mp  # import mediapipe for hand tracking
import pyautogui  # import pyautogui for controlling powerpoint
import time  # import time module for time-related operations

# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # initialize hands tracker with max 1 hand
mp_draw = mp.solutions.drawing_utils  # utility functions for drawing landmarks

# initialize opencv video capture
cap = cv2.VideoCapture(0)  # capture video from default camera (index 0)

# constants for box dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BOX_WIDTH = 150
BOX_HEIGHT = 480

LEFT_BOX = (0, 0, BOX_WIDTH, BOX_HEIGHT)  # dimensions of the left box
RIGHT_BOX = (FRAME_WIDTH - BOX_WIDTH, 0, BOX_WIDTH, BOX_HEIGHT)  # dimensions of the right box

# cooldown time in seconds
COOLDOWN_TIME = 1.0
last_action_time = time.time()

# function to control powerpoint based on gestures
def control_powerpoint(direction):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time > COOLDOWN_TIME:
        if direction == "left":
            pyautogui.press('left')  # simulate left arrow key press with pyautogui
            print("hand in left box - moving to previous slide")
        elif direction == "right":
            pyautogui.press('right')  # simulate right arrow key press with pyautogui
            print("hand in right box - moving to next slide")
        last_action_time = current_time

# main function to process video feed and detect gestures
def main():
    cap = cv2.VideoCapture(0)  # open default camera
    if not cap.isOpened():
        print("error: failed to capture video.")
        return
    
    while True:
        ret, frame = cap.read()  # read a frame from the camera
        if not ret:
            print("error: failed to capture frame from camera.")
            break
        
        frame = cv2.flip(frame, 1)  # flip the frame horizontally for natural viewing
        
        # draw boxes on the frame for gesture detection areas
        cv2.rectangle(frame, LEFT_BOX[:2], (LEFT_BOX[0] + LEFT_BOX[2], LEFT_BOX[1] + LEFT_BOX[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, RIGHT_BOX[:2], (RIGHT_BOX[0] + RIGHT_BOX[2], RIGHT_BOX[1] + RIGHT_BOX[3]), (0, 255, 0), 2)
        
        # convert frame to rgb for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # draw landmarks and connections
                
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * frame.shape[1])  # x coordinate of the landmark in the frame
                    y = int(lm.y * frame.shape[0])  # y coordinate of the landmark in the frame
                    
                    # check if hand is in the left box
                    if LEFT_BOX[0] <= x <= LEFT_BOX[0] + LEFT_BOX[2] and LEFT_BOX[1] <= y <= LEFT_BOX[1] + LEFT_BOX[3]:
                        control_powerpoint("left")  # call function to control powerpoint for left gesture
                        break
                    # check if hand is in the right box
                    elif RIGHT_BOX[0] <= x <= RIGHT_BOX[0] + RIGHT_BOX[2] and RIGHT_BOX[1] <= y <= RIGHT_BOX[1] + RIGHT_BOX[3]:
                        control_powerpoint("right")  # call function to control powerpoint for right gesture
                        break
        
        # display the frame with annotations
        cv2.imshow('gesture control', frame)
        
        # exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release the camera and close opencv windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
