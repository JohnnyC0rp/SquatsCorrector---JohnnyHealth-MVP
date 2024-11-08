import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from body_points import *
from funcs import *
import math
import pygame as pg
from random import choice
from time import sleep

pg.mixer.init()
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()
pose = mp_pose.Pose(
    min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=0
)


cap = cv2.VideoCapture(1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

play_audio("audio\\start.wav")
input()


ax.view_init(elev=-90, azim=-90)
frame_number = 0
coords_offset = -50
coords_offset_y = 850
axes_scale_multiplier = 0.5
wrong_angle_cnt = 0
knee_out_of_ankle_cnt = 0
back_angle_cnt = 0
progress_head = None
progress_cnt = 0
repetitions = 0
playing_correction = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    aspect_ratio = frame.shape[1] / frame.shape[0]
    # frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ax.clear()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([0, frame.shape[1] * axes_scale_multiplier])
    ax.set_ylim([0, frame.shape[0] * axes_scale_multiplier])
    ax.set_zlim([-2, 2])

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle_left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle_right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Draw 3D pose landmarks and bones
    if pose_results.pose_landmarks:

        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )  # draw on video

        for landmark in pose_results.pose_landmarks.landmark:
            landmark.x = landmark.x * frame.shape[1] + coords_offset
            landmark.y = landmark.y * frame.shape[0] + coords_offset + coords_offset_y
            ax.scatter(
                landmark.x,
                landmark.y / aspect_ratio - frame.shape[0] * aspect_ratio,
                landmark.z,
                c="#00fffb",
                marker="o",
            )
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_results.pose_landmarks.landmark[start_idx]
            end = pose_results.pose_landmarks.landmark[end_idx]

            if start_idx in upper_body_landmarks and end_idx in upper_body_landmarks:
                color = arm_color
            elif start_idx in lower_body_landmarks and end_idx in lower_body_landmarks:
                color = leg_color
            else:
                color = body_color

            ax.plot(
                [start.x, end.x],
                [
                    start.y / aspect_ratio - frame.shape[0] * aspect_ratio,
                    end.y / aspect_ratio - frame.shape[0] * aspect_ratio,
                ],
                [start.z, end.z],
                c=color,
            )

        head_y = 1100 - landmarks[1].y
        print(frame.shape, landmarks[1].y)

        if not progress_head:
            highest_head = head_y
            progress_head = True
            cur_state = "high"

        if (cur_state == "low") and ((head_y / highest_head) > 0.8):
            cur_state = "high"
            repetitions += 1
            # if not playing_correction:
            play_audio_count(f"audio\\{repetitions}.wav")
            if repetitions == 3 or repetitions == 7:
                (2)
                play_audio("audio\\random_complement.wav")
            if repetitions == 10:
                play_audio("audio\\complete.wav")
                input()
                exit()

        if (cur_state == "high") and ((head_y / highest_head) < 0.5):
            cur_state = "low"

        print(
            cur_state,
            "=====",
            head_y / highest_head,
            highest_head,
            head_y,
        )

        # ========= Checking different criteria =========

        # --------- knees angles ---------

        left_knee_angle = calculate_angle_2d(
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
        )
        right_knee_angle = calculate_angle_2d(
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
        )

        left_knee = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]

        avg_angle = (left_knee_angle + right_knee_angle) / 2

        cv2.putText(
            image,
            f"knees angle |||{round(float(avg_angle), 2)} deg",
            (int(left_knee.x - coords_offset), int(left_knee.y - coords_offset_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),  # Green text
            1,
            cv2.LINE_AA,
        )

        if avg_angle < 70:
            wrong_angle_cnt += 1
            if wrong_angle_cnt > 30:
                playing_correction = True
                play_audio(f"audio\\{choice(['wrong_angle.wav', 'wrong_angle2.wav'])}")
                wrong_angle_cnt = 0

                playing_correction = False

        # --------- knees out of ankle ------------

        avg_knees = (left_knee.x + right_knee.x) / 2
        avg_ankles = (sum([i.x for i in landmarks[27:33]])) / 6

        cv2.putText(
            image,
            f"knees out of ankle |||{round(avg_knees, 2)} ||| {round(avg_ankles, 2)}",
            (
                int(ankle_left.x - coords_offset - 200),
                int(ankle_left.y - coords_offset - coords_offset_y),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),  # Green text
            1,
            cv2.LINE_AA,
        )

        if avg_knees > (avg_ankles + 20):
            knee_out_of_ankle_cnt += 1
            if knee_out_of_ankle_cnt > 30:
                playing_correction = True
                play_audio(f"audio\\{choice(['wrong_dst.wav', 'wrong_dst2.wav'])}")
                knee_out_of_ankle_cnt = 0

                playing_correction = False

        # --------- back angle ---------

        back_angle = calculate_angle_2d(landmarks[12], landmarks[24], landmarks[26])
        print("===", back_angle)

        # if back_angle > 165 or back_angle < 100:
        #     back_angle_cnt += 1
        #     if back_angle_cnt > 10:
        #         play_audio(f"audio\\{choice(['back.wav', 'back2.wav'])}")
        #         back_angle_cnt = 0

        cv2.putText(
            image,
            f"Back angle: ||| {round(back_angle, 2)} |||",
            (
                int(landmarks[24].x - coords_offset),
                int(landmarks[24].y - coords_offset - coords_offset_y),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),  # Green text
            1,
            cv2.LINE_AA,
        )

    plt.draw()

    cv2.imshow("Pose and Hand Detection", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        frame_number += 1  # Next frame ➡️
    elif key == ord("p") and frame_number > 0:
        frame_number -= 1  # Previous frame ⬅️
    elif key == ord("q"):
        break  # Quit 🚪


cap.release()
cv2.destroyAllWindows()


# when repitiotions equal 10 play compliemnt
