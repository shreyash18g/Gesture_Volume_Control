import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing_hands = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open the default camera
cap = cv2.VideoCapture(0)

# Set camera resolution
w_cam, h_cam = 1280, 720
cap.set(3, w_cam)
cap.set(4, h_cam)

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Initial volume and bar settings
vol = 0
vol_bar = 600
vol_per = 0

# Initialize previous time for FPS calculation
p_time = 0

# Main loop for processing frames
while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture frame.")
        break
    
    # Convert the image to RGB format for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect hands in the image
    results = hands.process(img_rgb)
    
    # Process each detected hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw hand landmarks on the frame
            mp_drawing_hands.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract thumb and index finger tip landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert landmarks to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * w_cam), int(thumb_tip.y * h_cam)
            index_x, index_y = int(index_tip.x * w_cam), int(index_tip.y * h_cam)
            
            # Calculate the distance between thumb and index finger tips
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Map distance to volume range
            vol = np.interp(distance, [50, 300], [min_vol, max_vol])
            vol_bar = np.interp(distance, [50, 300], [600, 150])
            vol_per = np.interp(distance, [50, 300], [0, 100])
            
            # Set system volume
            volume.SetMasterVolumeLevel(vol, None)
            
            # Determine line color based on volume percentage
            if vol_per <= 50:
                line_color = (0, 255, 0)  # Green for normal volume
            elif vol_per <= 80:
                line_color = (255, 0, 0)  # Blue for high volume
            else:
                line_color = (0, 0, 255)  # Red for warning level volume
            
            # Draw line between thumb and index finger tips
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), line_color, 4)
            
            # Draw circles on thumb tip, index finger tip, and midpoint
            cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            mid_x, mid_y = int((thumb_x + index_x) / 2), int((thumb_y + index_y) / 2)
            cv2.circle(img, (mid_x, mid_y), 10, (0, 255, 255), cv2.FILLED)
    
    # Draw volume bar background
    cv2.rectangle(img, (50, 150), (85, 600), (0, 0, 0), cv2.FILLED)
    
    # Draw volume bar and add warning text based on volume percentage
    if vol_per <= 50:
        cv2.rectangle(img, (50, int(vol_bar)), (85, 600), (0, 255, 0), cv2.FILLED)  # Green volume bar
        cv2.putText(img, "Normal Volume", (50, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)  # Warning message
    elif vol_per <= 80:
        cv2.rectangle(img, (50, int(vol_bar)), (85, 600), (255, 0, 0), cv2.FILLED)  # Blue volume bar
        cv2.putText(img, "High Volume", (50, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)  # Warning message
    else:
        cv2.rectangle(img, (50, int(vol_bar)), (85, 600), (0, 0, 255), cv2.FILLED)  # Red volume bar
        cv2.putText(img, "WARNING: Volume too high!", (50, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)  # Warning message
    
    # Display volume percentage
    cv2.putText(img, f'{int(vol_per)}%', (40, 650), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)  # White text
    
    # Calculate and display FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)  # White text
    
    # Show the frame
    cv2.imshow('Hand Volume Control', img)
    
    # Check for exit key (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
