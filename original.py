import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()
pose = mp_pose.Pose(
    min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=0
)

cap = cv2.VideoCapture("test1.mp4")

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.view_init(elev=-90, azim=-90)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image)
    # hand_results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ax.clear()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Draw 3D pose landmarks and bones
    if pose_results.pose_landmarks:
        # mp_drawing.draw_landmarks(
        #     image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        # )
        for landmark in pose_results.pose_landmarks.landmark:
            ax.scatter(landmark.x, landmark.y, landmark.z, c="r", marker="o")
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_results.pose_landmarks.landmark[start_idx]
            end = pose_results.pose_landmarks.landmark[end_idx]
            ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], c="r")

    # # Draw 3D hand landmarks and bones
    # if hand_results.multi_hand_landmarks:
    #     for hand_landmarks in hand_results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #         for landmark in hand_landmarks.landmark:
    #             ax.scatter(landmark.x, landmark.y, landmark.z, c="b", marker="o")
    #         for connection in mp_hands.HAND_CONNECTIONS:
    #             start_idx, end_idx = connection
    #             start = hand_landmarks.landmark[start_idx]
    #             end = hand_landmarks.landmark[end_idx]
    #             ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], c="b")

    plt.draw()
    # plt.pause(0.001)

    cv2.imshow("Pose and Hand Detection", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
