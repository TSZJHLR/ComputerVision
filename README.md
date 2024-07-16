# Computer Vision and Gesture Control Scripts

## 1. art.py

**Description:** This script allows users to paint on a canvas using hand gestures captured via a webcam. It detects various gestures such as pinch for drawing, fist for erasing, and open hand for changing colors.

**Features:**
- Real-time hand gesture detection using Mediapipe Hands.
- Canvas drawing functionalities with adjustable brush sizes and colors.
- Gesture-based controls for drawing, erasing, and color changing.

---

## 2. pptx.py

**Description:** This script enables gesture-based control of PowerPoint slides using hand movements captured via a webcam. It detects hand positions in predefined boxes to navigate between slides.

**Features:**
- Navigation through PowerPoint slides using left and right hand positions.
- Integration of Mediapipe for real-time hand landmark detection.
- Simple and intuitive control mechanism using predefined gesture areas.

---

## 3. prepro.py

**Description:** This script demonstrates basic image preprocessing techniques using OpenCV and Pillow (PIL) libraries. It processes a region of interest (ROI) extracted from a webcam feed, applying operations like grayscale conversion, Gaussian blur, skin detection, and denoising.

**Features:**
- Image preprocessing operations using OpenCV and Pillow.
- Visualization of various stages of image processing (grayscale, blurred, thresholded, denoised, RGB channels, and skin mask).
- Real-time processing of video feed from the webcam.

---

## 4. volume.py

**Description:** This script adjusts the system volume based on the distance between thumb and index finger gestures captured via a webcam. It utilizes Mediapipe for hand landmark detection and Pycaw for system volume control.

**Features:**
- Dynamic adjustment of system volume based on hand gestures.
- Visual feedback of current volume level and FPS (frames per second) on the video feed.
- Real-time volume level updates and visual representation of volume control.

---

## Installation

### Prerequisites

- Python 3.x installed on your system
- Pip package manager (usually comes with Python installation)

### Dependencies

Install the required Python packages using pip:

```bash
pip install opencv-python mediapipe pycaw comtypes Pillow numpy
