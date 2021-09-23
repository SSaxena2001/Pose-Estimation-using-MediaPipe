import cv2
import mediapipe as mp
import time
from PoseEstimationModule import PoseDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lmList = detector.get_position(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        resize = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Image", resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    main()