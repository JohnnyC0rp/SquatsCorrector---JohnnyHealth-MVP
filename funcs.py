import numpy as np
import pygame


def calculate_angle(p1, p2, p3):
    # Calculate angle between points p1 -> p2 -> p3
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)


def calculate_angle_2d(p1, p2, p3):
    # Calculate angle between points p1 -> p2 -> p3 using x and y
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)


def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def play_audio_count(file_path):
    sound_effect = pygame.mixer.Sound(file_path)
    sound_effect.play()  # Play a sound effect while music is on
