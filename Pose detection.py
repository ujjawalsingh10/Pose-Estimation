import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cap = cv2.VideoCapture('Videos/1.mp4')

def rescaleFrame(frame, scale = 0.2):  #Frame is the frame at that point
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (height, width)  #set tuple with the new dimension
    return cv2.resize(frame, dimensions,  interpolation=cv2.INTER_AREA)

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  #mpPose.POSE_CONNECTIONS connects all the landmarks and draws lines
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx,cy), 2, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    frame_resized = rescaleFrame(frame, scale = 0.6) 
    cv2.imshow('Video', frame_resized)
    
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
