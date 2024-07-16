import cv2  # import the OpenCV library for computer vision tasks
import numpy as np  # import numpy for numerical operations
from PIL import Image, ImageOps, ImageFilter  # import necessary modules from PIL (Pillow) for image manipulation

# initialize OpenCV video capture using the default camera (0)
cap = cv2.VideoCapture(0)

# constants for defining dimensions of the region of interest (ROI)
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # get width of the video frame
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # get height of the video frame
BOX_WIDTH = FRAME_WIDTH // 3  # width of the ROI box is one-third of the frame width
BOX_HEIGHT = FRAME_HEIGHT // 3  # height of the ROI box is one-third of the frame height
BOX_X = (FRAME_WIDTH - BOX_WIDTH) // 2  # x-coordinate of the top-left corner of the ROI box
BOX_Y = (FRAME_HEIGHT - BOX_HEIGHT) // 2  # y-coordinate of the top-left corner of the ROI box

# function to preprocess the region of interest (ROI) using Pillow (PIL) for image operations
def preprocess_frame(roi):
    results = {}  # initialize an empty dictionary to store preprocessing results
    
    # convert the ROI to grayscale using Pillow's ImageOps
    gray = ImageOps.grayscale(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
    results["gray"] = np.array(gray)  # convert the grayscale image to a numpy array and store in results
    
    # apply Gaussian blur to the ROI using Pillow's ImageFilter (demonstration purpose)
    blurred = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).filter(ImageFilter.GaussianBlur(radius=5))
    results["blurred"] = np.array(blurred)  # convert the blurred image to a numpy array and store in results
    
    # convert the blurred image back to OpenCV's BGR format
    results["blurred"] = cv2.cvtColor(results["blurred"], cv2.COLOR_RGB2BGR)
    
    # convert the ROI to RGB color space using Pillow's Image
    rgb_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    r, g, b = rgb_image.split()  # split the RGB channels
    results["r"] = np.array(r)  # convert the red channel to a numpy array and store in results
    results["g"] = np.array(g)  # convert the green channel to a numpy array and store in results
    results["b"] = np.array(b)  # convert the blue channel to a numpy array and store in results
    
    # perform skin detection based on RGB color space using numpy operations
    r_np = np.array(r)  # convert the red channel back to a numpy array
    g_np = np.array(g)  # convert the green channel back to a numpy array
    b_np = np.array(b)  # convert the blue channel back to a numpy array
    
    skin_mask = np.zeros_like(r_np, dtype=np.uint8)  # initialize a binary mask for skin detection
    
    # define conditions for skin detection based on RGB values
    skin_condition = (r_np > 95) & (g_np > 40) & (b_np > 20) & \
                     ((np.maximum(r_np, np.maximum(g_np, b_np)) - np.minimum(r_np, np.minimum(g_np, b_np))) > 15) & \
                     (np.abs(r_np - g_np) > 15) & (r_np > g_np) & (r_np > b_np)
    
    skin_mask[skin_condition] = 255  # set pixels that satisfy the skin conditions to 255 (white)
    results["skin mask"] = skin_mask  # store the skin mask in results
    
    # apply thresholding to the grayscale image using Pillow (demonstration purpose)
    thresholded = gray.point(lambda p: p > 127 and 255)  # threshold pixels in the grayscale image
    results["thresholded"] = np.array(thresholded)  # convert the thresholded image to a numpy array and store in results
    
    return results  # return the dictionary containing preprocessing results

# main function to process video feed and display preprocessing results
def main():
    if not cap.isOpened():  # check if the camera is opened successfully
        print("Error: Failed to capture video.")  # print error message if failed to capture video
        return  # exit the function
    
    # create windows for each preprocessing result using OpenCV
    cv2.namedWindow("gray")
    cv2.namedWindow("blurred")
    cv2.namedWindow("thresholded")
    cv2.namedWindow("r")
    cv2.namedWindow("g")
    cv2.namedWindow("b")
    cv2.namedWindow("skin mask")
    
    while True:  # loop indefinitely to process each frame
        ret, frame = cap.read()  # read a frame from the video capture
        if not ret:  # if frame reading fails
            print("Error: Failed to capture frame from camera.")  # print error message
            break  # exit the loop
        
        frame = cv2.flip(frame, 1)  # flip the frame horizontally for natural viewing
        
        # draw a box on the frame to indicate the ROI
        cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_WIDTH, BOX_Y + BOX_HEIGHT), (0, 255, 0), 2)
        
        # extract the region of interest (ROI) from the frame
        roi = frame[BOX_Y:BOX_Y + BOX_HEIGHT, BOX_X:BOX_X + BOX_WIDTH]
        
        # apply preprocessing to the ROI using preprocess_frame function
        preprocessed = preprocess_frame(roi)
        
        # display the preprocessing results in their respective windows using OpenCV's imshow function
        cv2.imshow("gray", preprocessed["gray"])
        cv2.imshow("blurred", preprocessed["blurred"])
        cv2.imshow("thresholded", preprocessed["thresholded"])
        cv2.imshow("r", preprocessed["r"])
        cv2.imshow("g", preprocessed["g"])
        cv2.imshow("b", preprocessed["b"])
        cv2.imshow("skin mask", preprocessed["skin mask"])
        
        # display the original frame with annotations
        cv2.imshow('gesture control', frame)
        
        # exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # release the camera
    cv2.destroyAllWindows()  # close all OpenCV windows

if __name__ == "__main__":
    main()  # call the main function if the script is executed directly
