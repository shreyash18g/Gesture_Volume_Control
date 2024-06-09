# Hand Gesture Volume Control

Hand Gesture Volume Control is a Python application that uses the MediaPipe library to detect hand gestures and control system volume based on the distance between the thumb and index finger.

## Description

The application uses the MediaPipe Hands module to detect hand landmarks from the webcam feed. It calculates the distance between the thumb and index finger tips to control the system volume.

## Features

- **Real-time Hand Gesture Detection**: Utilizes MediaPipe Hands to detect hand gestures.
- **Volume Control**: Adjusts the system volume based on hand gestures.
- **Visual Feedback**: Displays the volume bar and percentage on the screen.
- **Dynamic Warning**: Provides warning messages for high volume levels.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- Numpy
- Pycaw

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/shreyash18g/Gesture_Volume_Control.git
    cd Gesture_Volume_Control
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python mediapipe numpy pycaw
    ```

3. Run the application:

    ```bash
    python Gesture_Volume_Control.py
    ```

4. Adjust system volume by bringing your thumb and index finger closer or farther apart.

5. Press 'q' to exit the application.

## How it Works

The application captures frames from the webcam and uses MediaPipe Hands to detect hand landmarks. It calculates the distance between the thumb and index finger tips and maps it to the system volume range. Visual feedback is provided on the screen, along with warning messages for high volume levels.

## Contributing

Contributions to this project are welcome! Feel free to open issues or submit pull requests.


## Acknowledgments

- This project utilizes the MediaPipe library for hand gesture detection.
- Special thanks to the contributors of OpenCV, Numpy, and Pycaw libraries.
