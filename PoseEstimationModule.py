import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self,
                 mode=False,
                 model_complex=1,
                 smooth=True,
                 enable_seg=False,
                 smooth_seg=True,
                 detection_con=0.5,
                 tracking_con=0.5):
        self.mode = mode
        self.model_complex = model_complex
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complex, self.smooth, self.enable_seg, self.smooth_seg, self.detection_con, self.tracking_con)
        self.mpDraw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def get_position(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                   cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList