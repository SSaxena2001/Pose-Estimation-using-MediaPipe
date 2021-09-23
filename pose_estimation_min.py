import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("PoseVideos/2.mp4")
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    resize = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
