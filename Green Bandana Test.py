import mediapipe as mp
import cv2
import numpy as np




mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
file_path = "images.h5"
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks 

cap = cv2.VideoCapture(0)

def getFaceRect(landmarks, enlarge):
    face_top = int(landmarks[10].y * image.shape[0])- enlarge
    face_left = int(landmarks[234].x * image.shape[1]) - enlarge
    face_bottom = int(landmarks[152].y * image.shape[0])+enlarge
    face_right = int(landmarks[454].x * image.shape[1]) +enlarge

    h = face_bottom - face_top
    w = face_right - face_left

    x = face_left
    y = face_top
    return x, y, w, h

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make face detection
        results = face_mesh.process(image)
        face_landmarks = results.multi_face_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if face_landmarks:
            print(face_landmarks)
            face_landmark = face_landmarks[0]
            xFace, yFace, faceWidth, faceHeight = getFaceRect(face_landmark.landmark, 50)
            faceROI = image[yFace:yFace + faceHeight, xFace:xFace+faceWidth].copy()
            face_hsv = cv2.cvtColor(faceROI, cv2.COLOR_BGR2HSV)
            
            # Create a mask to segment the green color within the face ROI
            face_mask = cv2.inRange(face_hsv, lower_green, upper_green)
            
            # Calculate the percentage of green pixels in the face ROI
            green_pixel_percentage = (cv2.countNonZero(face_mask) / (faceWidth * faceHeight)) * 100
            if green_pixel_percentage > 5:  # Adjust this threshold as needed
                cv2.rectangle(frame, (xFace, yFace), (xFace + faceWidth, yFace + faceHeight), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (xFace, yFace), (xFace + faceWidth, yFace + faceHeight), (0, 0, 255), 2)
        cv2.imshow('Bandana Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
