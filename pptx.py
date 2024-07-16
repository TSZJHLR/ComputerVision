import cv2
import numpy as np
import pyautogui

# Constants for gesture detection
GESTURE_THRESHOLD = 50  # Minimum pixels for a gesture to be detected

# Function to control PowerPoint based on gestures
def control_powerpoint(direction):
    if direction == "left":
        pyautogui.hotkey('left')  # Simulate left arrow key press
        print("Swipe left detected - Moving to previous slide")
    elif direction == "right":
        pyautogui.hotkey('right')  # Simulate right arrow key press
        print("Swipe right detected - Moving to next slide")

# Main function to process video feed and detect gestures
def main():
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Failed to capture video.")
        return
    
    # Variables for gesture detection
    start_x = None
    gesture_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break
        
        # Flip frame horizontally for natural viewing
        frame = cv2.flip(frame, 1)
        
        # Convert frame to grayscale for simplicity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Smooth the image to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edged image
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour, which should be the hand
            contour = max(contours, key=cv2.contourArea)
            
            # Ensure the contour area is large enough to be considered a hand
            if cv2.contourArea(contour) > GESTURE_THRESHOLD:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw the bounding box around the detected hand
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate the centroid of the hand
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                
                # Check for gesture start (left edge of hand)
                if start_x is None:
                    start_x = centroid_x
                
                # Detect left or right swipe based on hand movement
                if centroid_x < start_x - GESTURE_THRESHOLD:
                    control_powerpoint("left")
                    start_x = centroid_x  # Reset start position for next gesture detection
                    gesture_detected = True
                elif centroid_x > start_x + GESTURE_THRESHOLD:
                    control_powerpoint("right")
                    start_x = centroid_x  # Reset start position for next gesture detection
                    gesture_detected = True
        
        # Display the frame with annotations
        cv2.imshow('Gesture Control', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
