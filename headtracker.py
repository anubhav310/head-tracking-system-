import cv2
import time
import webbrowser
import numpy as np
from collections import deque

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading face detection model. Make sure OpenCV is installed correctly.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Button positions and states
buttons = {
    "portfolio": {"x1": 50, "y1": 50, "x2": 250, "y2": 150, "hover_time": 0},
    "resume": {"x1": 350, "y1": 50, "x2": 550, "y2": 150, "stable_time": 0}
}

# Head position history for stability detection
head_positions = deque(maxlen=15)

# Replace with your actual URLs
URLS = {
    "portfolio": "linki ",
    "resume": "https://drive.google.com/file/d/1CHIg5azbV3GgRtjcXe1BOQfLbAPrqQmQ/view?usp=sharing"
}

# Main loop
last_action_time = time.time()
cooldown = 3  # seconds between actions

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100))
    
    # Draw buttons
    cv2.rectangle(frame, (buttons["portfolio"]["x1"], buttons["portfolio"]["y1"]), 
                 (buttons["portfolio"]["x2"], buttons["portfolio"]["y2"]), (0, 200, 0), 2)
    cv2.putText(frame, "PORTFOLIO", (buttons["portfolio"]["x1"] + 20, buttons["portfolio"]["y1"] + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.rectangle(frame, (buttons["resume"]["x1"], buttons["resume"]["y1"]), 
                 (buttons["resume"]["x2"], buttons["resume"]["y2"]), (0, 100, 200), 2)
    cv2.putText(frame, "RESUME", (buttons["resume"]["x1"] + 50, buttons["resume"]["y1"] + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
    
    head_center = None
    
    if len(faces) > 0:
        # Get the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        # Calculate head center
        head_center = (x + w//2, y + h//2)
        
        # Draw face rectangle and center point
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, head_center, 8, (0, 255, 255), -1)
        cv2.circle(frame, head_center, 10, (0, 100, 255), 2)
        
        # Store head position
        head_positions.append(head_center)
    
    # Button interactions
    current_time = time.time()
    if head_center:
        x, y = head_center
        
        # Portfolio button (hover for 5 seconds)
        if (buttons["portfolio"]["x1"] <= x <= buttons["portfolio"]["x2"] and 
            buttons["portfolio"]["y1"] <= y <= buttons["portfolio"]["y2"]):
            buttons["portfolio"]["hover_time"] += 0.05  # Increment time
            
            # Draw progress bar
            progress = min(1.0, buttons["portfolio"]["hover_time"] / 5.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (buttons["portfolio"]["x1"], buttons["portfolio"]["y2"] + 5), 
                         (buttons["portfolio"]["x1"] + bar_width, buttons["portfolio"]["y2"] + 15), 
                         (0, 255, 0), -1)
            
            # Draw time text
            time_left = max(0, 5 - buttons["portfolio"]["hover_time"])
            cv2.putText(frame, f"{time_left:.1f}s", (buttons["portfolio"]["x1"] + 90, buttons["portfolio"]["y2"] + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Trigger at 5 seconds with cooldown
            if buttons["portfolio"]["hover_time"] >= 5 and current_time - last_action_time > cooldown:
                webbrowser.open(URLS["portfolio"])
                buttons["portfolio"]["hover_time"] = 0
                last_action_time = current_time
        else:
            buttons["portfolio"]["hover_time"] = 0
        
        # Resume button (stabilize head)
        if (buttons["resume"]["x1"] <= x <= buttons["resume"]["x2"] and 
            buttons["resume"]["y1"] <= y <= buttons["resume"]["y2"]):
            
            # Check stability
            stable = False
            if len(head_positions) > 5:
                # Calculate movement variance
                positions = np.array(head_positions)
                distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
                avg_movement = np.mean(distances)
                stable = avg_movement < 5  # Threshold for stability
            
            if stable:
                buttons["resume"]["stable_time"] += 0.05
                # Draw stability indicator
                cv2.circle(frame, (buttons["resume"]["x2"] - 20, buttons["resume"]["y1"] + 20), 
                          8, (0, 255, 0), -1)
                
                # Draw stability progress
                cv2.putText(frame, f"Stable: {buttons['resume']['stable_time']:.1f}s", 
                           (buttons["resume"]["x1"], buttons["resume"]["y2"] + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Trigger after 1 second stable
                if buttons["resume"]["stable_time"] >= 1 and current_time - last_action_time > cooldown:
                    webbrowser.open(URLS["resume"])
                    buttons["resume"]["stable_time"] = 0
                    last_action_time = current_time
            else:
                buttons["resume"]["stable_time"] = 0
        else:
            buttons["resume"]["stable_time"] = 0
    
    # Display instructions
    cv2.putText(frame, "Hover over PORTFOLIO for 5s to open portfolio", 
               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    cv2.putText(frame, "Stabilize head over RESUME to open resume", 
               (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    cv2.putText(frame, "Press 'Q' to exit", (width - 200, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
    
    # Show frame
    cv2.imshow('Head Tracking System', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()