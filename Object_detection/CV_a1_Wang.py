# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 02:08:41 2023

@author: zhiqi
"""

import numpy as np
import cv2

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg.txt")
classes = []
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the minimum confidence level and non-maximum suppression threshold
confThreshold = 0.05
nmsThreshold = 0.3

# Load the Haar Cascade Classifier for upper body
hand_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

# Load the Haar Cascade Classifier for eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

def apply_effect(frame, time, cap, width, height):
    #frame = cv2.flip(frame, 0)  # Add this line to flip the frame vertically
    frame_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))


#%% 
# 1.Basic image processing (0-20s)    
 #1.1 Switch color
    if 0 <= frame_time_ms < 5000:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (frame_time_ms // 500) % 2 == 0 else frame, "Switching between color and grayscale"
        
        subtitle = f"1.1 Switch beween color and grayscale"

 #1.2 Blurring (Gaussian & Bi-lateral)  
    elif 5 <= time < 13:
        kernel_step = (time - 5) // 2
        kernel_size = min(1 + 4 * kernel_step, 13)
        if time % 2 == 0:
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0), f"1.2 Gaussian Blur (k = {kernel_size}):\nSmoother image, blurred edges"
        else:
            return cv2.bilateralFilter(frame, kernel_size, 75, 75), f"1.2 Bi-lateral Filter (k = {kernel_size}):\nSmoother image, keeps edges sharp"

 #1.3 Grab banana in RGB and HSV color space, then improve grabbing in HSV space
 
         # 1.3.1 In HSV color space
    elif 13 <= time < 15:
        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds of the banana color
        lower_banana_color = np.array([20, 100, 100])
        upper_banana_color = np.array([30, 255, 255])

        # Create a mask for the banana color in HSV space
        banana_mask_hsv = cv2.inRange(hsv, lower_banana_color, upper_banana_color)
        
        # Create a black and white image of the banana in HSV space
        banana_bw_hsv = cv2.bitwise_and(frame, frame, mask=banana_mask_hsv)
        banana_bw_hsv[banana_mask_hsv == 0] = [0, 0, 0]
        banana_bw_hsv = cv2.cvtColor(banana_bw_hsv, cv2.COLOR_BGR2GRAY)
        _, banana_bw_hsv = cv2.threshold(banana_bw_hsv, 0, 255, cv2.THRESH_BINARY)

        return banana_bw_hsv, "1.3 Object Grabbing in HSV color space"

        # 1.3.2 In RBG color space
    elif 15 <= time < 17:
        # Define the lower and upper bounds of the banana color in RGB space
        lower_banana_color_rgb = np.array([0, 70, 140])
        upper_banana_color_rgb = np.array([60, 255, 255])

        # Create a mask for the banana color in RGB space
        banana_mask_rgb = cv2.inRange(frame, lower_banana_color_rgb, upper_banana_color_rgb)
       
        # Create a black and white image of the banana in RGB space
        banana_bw_rgb = cv2.bitwise_and(frame, frame, mask=banana_mask_rgb)
        banana_bw_rgb[banana_mask_rgb == 0] = [0, 0, 0]
        banana_bw_rgb = cv2.cvtColor(banana_bw_rgb, cv2.COLOR_BGR2GRAY)
        _, banana_bw_rgb = cv2.threshold(banana_bw_rgb, 0, 255, cv2.THRESH_BINARY)

        return banana_bw_rgb, "1.3 Object Grabbing in RGB color space"
 
        # 1.3.3 Improve in RGB color space
    elif 17 <= time < 20:    
        # Initialize variables
        banana_bw_rgb = None
        banana_bw_hsv = None
        #banana_bw_combined = None

        # Define the lower and upper bounds of the banana color
        lower_banana_color_rgb = np.array([0, 70, 140])
        upper_banana_color_rgb = np.array([60, 255, 255])

        # Create a mask for the banana color in RGB space
        banana_mask_rgb = cv2.inRange(frame, lower_banana_color_rgb, upper_banana_color_rgb)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((2, 2), np.uint8)
        banana_mask_rgb = cv2.morphologyEx(banana_mask_rgb, cv2.MORPH_OPEN, kernel)
        banana_mask_rgb = cv2.morphologyEx(banana_mask_rgb, cv2.MORPH_CLOSE, kernel)
        banana_mask_rgb = cv2.erode(banana_mask_rgb, kernel, iterations=1)

        # Fill holes in the mask using a binary morphological operation
        contours, _ = cv2.findContours(banana_mask_rgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and draw them on the mask
        min_area = 50
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(banana_mask_rgb, [contour], 0, 255, -1)

        # Create a black and white image of the banana with the improved mask
        banana_bw_improved = cv2.bitwise_and(frame, frame, mask=banana_mask_rgb)
        banana_bw_improved = cv2.cvtColor(banana_bw_improved, cv2.COLOR_BGR2GRAY)
        banana_bw_improved = cv2.merge((np.zeros_like(banana_bw_improved), np.zeros_like(banana_bw_improved), cv2.threshold(banana_bw_improved, 0, 255, cv2.THRESH_BINARY)[1]))
 
        return banana_bw_improved, "1.3 Improved Object Grabbing in RGB color space:\nFill holes"
 
#%%
    
# 2. Object detection (20-40s)
 
 # 2.1 Sobel edge detection
    elif 20 <= time < 25:
    # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale_factor = 1
        if 20 <= time < 21:
            ksize = 1
        elif 21 <= time < 22:
            ksize = 5
        elif 22 <= time < 23:
            scale_factor = 2
            ksize = 1
        else:
            scale_factor = 2
            ksize = 5

    # Apply Sobel edge detection for horizontal and vertical edges
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calculate the absolute values of the gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

    # Normalize the gradients to the range 0-255
        scaled_sobelx = np.uint8(255 * scale_factor * abs_sobelx / np.max(abs_sobelx))
        scaled_sobely = np.uint8(255 * scale_factor * abs_sobely / np.max(abs_sobely))

    # Merge the gradients using different colors
        edge_visualization = cv2.merge((scaled_sobelx, np.zeros_like(scaled_sobelx), scaled_sobely))

        return edge_visualization, f"2.1 Sobel edge detection: \nVisualizing horizontal (blue) and vertical (red) edges, \nKernel size={ksize}, Scale factor={scale_factor}"

 # 2.2 Hough circle detection
    elif 25 <= time < 35:
    # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)

    # Apply Canny edge detection
        edges = cv2.Canny(blurred_frame, 45, 100)

    # Set Hough transform parameters
        dp = 1
        minDist = 20
        param1 = 30
        param2 = 20
        minRadius = 1
        maxRadius = 100

        if 27 <= time < 29:
            param2 = 30

        elif 29 <= time < 32:
            param2 = 30
            maxRadius = 70

        elif 32 <= time < 35:
            param2 = 30
            maxRadius = 70
            minDist = 60

        subtitle = f"2.2 Hough circle detection: \nWith param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}, minDist={minDist}"

    # Apply Hough transform to detect circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw detected circles on the original frame
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        return frame, subtitle

 # 2.3 Object introduce
    elif 35 <= time < 37:
    # Draw a flashy rectangle around the object (person's face)
        face_coordinates = (380, 160, 80, 80)
        color = (0, 255, 0) if time % 2 == 0 else (0, 0, 255)
        cv2.rectangle(frame, face_coordinates[:2], (face_coordinates[0]+face_coordinates[2], face_coordinates[1]+face_coordinates[3]), color, 2)

        subtitle = "2.3 Drawing a flashy rectangle around the person's face"

    elif 37 <= time < 40:
    # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create an intensity mask proportional to the likelihood of the object being at a particular location
    # Replace the center and sigma values with the actual face location and uncertainty values
        center = (415, 200)
        sigma = 30
        height, width = gray_frame.shape
        y = np.arange(0, height, 1)
        x = np.arange(0, width, 1)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma ** 2))

    # Normalize the intensity values to the range 0-255
        mask = np.uint8(255 * Z / Z.max())

    # Apply the intensity mask to the grayscale frame
        masked_frame = cv2.multiply(gray_frame, mask, scale=1 / 255)

        frame = masked_frame
        subtitle = "2.3 The likelihood of the face being at a particular location"


#%%

# 3. Carte blanche (40-60s)

 # 3.1 Arm detection and Tracking
    elif 40 <= time < 46:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect upper body using the Haar Cascade Classifier
        hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw bounding boxes around the detected hands
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        subtitle = "3.1 Arm Detection and Tracking: \nUsing Haar Cascade Classifier"
        
 # 3.2 Detect eyes with glasses (46-53s)
    elif 46 <= time < 53:
         scale_factor = 2
         resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
         gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
         eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    
         for (x, y, w, h) in eyes:
            x, y, w, h = x // scale_factor, y // scale_factor, w // scale_factor, h // scale_factor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

         subtitle = "3.2 Eye Detection with Glasses: \nUsing Haar Cascade Classifier"

 #3.3 Object Detection and Labeling (53-60s)
    elif 53 <= time < 60:
        # Get the width and height of the frame
        height, width, channels = frame.shape

        # Convert the frame to a blob and pass it through the network
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Get the class IDs, confidences, and bounding boxes
        classIds = []
        confidences = []
        boxes = []
        # Define the percentage to reduce the size of the boxes
        boxSizeReduction = 0.3  # 20%
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold and classId in [0, 21, 56]:
                    centerX = int(detection[0] * width)
                    centerY = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Apply non-maximum suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        # Draw the final boxes and labels on the frame
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                
                # Reduce the size of the box by a percentage
                wReduction = int(w * boxSizeReduction / 2)
                hReduction = int(h * boxSizeReduction / 2)
                x += wReduction
                y += hReduction
                w -= wReduction * 2
                h -= hReduction * 2

                label = f"{classes[classIds[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0) if classes[classIds[i]] == 'apple' else (0, 0, 255) if classes[classIds[i]] == 'banana' else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        subtitle = "3.3 Object Detection and Labeling: \nUsing YOLO"

    else:
        subtitle = ""

    return frame, subtitle


#%%                                      
def add_subtitle(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if isinstance(frame, np.ndarray):
        frame = cv2.UMat(frame)

        
    y_offset = 30
    line_spacing = 40
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (20, y_offset + i * line_spacing), font, 0.8, (255, 255, 255), 2)
    return frame.get()

input_video = "E:/OneDrive - KU Leuven/Desktop/CV2.mp4"
output_video = "E:/OneDrive - KU Leuven/Desktop/output.mp4"

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    time = int(cap.get(cv2.CAP_PROP_POS_MSEC)) // 1000
    frame, subtitle = apply_effect(frame, time,cap, width, height)
    frame = add_subtitle(frame, subtitle)

    out.write(frame)
    cv2.imshow('Output Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()