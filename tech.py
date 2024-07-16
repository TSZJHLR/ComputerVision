import cv2
import numpy as np

# Constants for image processing
GAUSSIAN_BLUR_KERNEL = (5, 5)
THRESHOLD_MIN = 100
THRESHOLD_MAX = 255

# Function to process each frame of the video feed
def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    
    # Apply adaptive thresholding
    _, thresh = cv2.threshold(blurred, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
    
    return gray, blurred, thresh

# Function to evaluate processed results and display them in windows on main frame
def evaluate_results(frame, gray, blurred, thresh):
    # Get dimensions of the frame
    height, width = frame.shape[:2]
    
    # Resize processed frames for display in small windows
    gray_resized = cv2.resize(gray, (width // 3, height // 3))
    blurred_resized = cv2.resize(blurred, (width // 3, height // 3))
    thresh_resized = cv2.resize(thresh, (width // 3, height // 3))
    
    # Create a black canvas to display processed frames
    canvas = np.zeros_like(frame)
    
    # Place resized frames on the canvas
    canvas[0:height//3, 0:width//3] = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    canvas[0:height//3, width//3:2*width//3] = cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR)
    canvas[0:height//3, 2*width//3:width] = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
    
    # Display the canvas with processed frames on the main frame
    cv2.imshow('Processed Frames', canvas)
    
    cv2.waitKey(1)

# Main function to capture video feed and apply processing
def main():
    cap = cv2.VideoCapture(0)  # Open default camera
    
    if not cap.isOpened():
        print("Error: Failed to capture video.")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break
        
        # Process each frame
        gray, blurred, thresh = process_frame(frame)
        
        # Evaluate and display results in small windows on the main frame
        evaluate_results(frame, gray, blurred, thresh)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
