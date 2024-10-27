import cv2
import random
import pygame
from deepface import DeepFace
import time
import pygetwindow as gw  # Import the pygetwindow library

# Initialize Pygame Mixer
pygame.mixer.init()

# Dummy music data based on emotions
emotion_to_music = {
    "happy": ["Up - Married Life.mp3"],
    "sad": ["Death-Bed-Powfu.mp3"],
    "neutral": ["128-Bade Achhe Lagte Hain - Balika Badhu 128 Kbps.mp3"]
}

# OpenCV Video Capture
cap = cv2.VideoCapture(0)  # Use 1 or 2 if 0 doesn't work

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
last_emotion = None
music_playing = False
emotion_start_time = None
emotion_stable_duration = 3  # Duration in seconds to consider the emotion stable
timer_display = False  # To control the visibility of the timer
timer_started = False  # Flag to ensure the timer only starts once
minimize_window_time = None  # Time to minimize the window after music starts

def play_music(emotion):
    global music_playing
    # Ensure the emotion is valid and available in the music dictionary
    if emotion in emotion_to_music:
        # Choose a random song for the detected emotion
        music_file = random.choice(emotion_to_music[emotion])
        
        # Load and play the new music
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(-1)  # -1 means the music will loop indefinitely
        music_playing = True  # Set flag to indicate music is playing

def reset_emotion_detection():
    global last_emotion, emotion_start_time, timer_display, timer_started, music_playing
    last_emotion = None
    emotion_start_time = None
    timer_display = False
    timer_started = False
    music_playing = False
    pygame.mixer.music.stop()  # Stop any currently playing music

# Create a named window
cv2.namedWindow('Real-time Emotion Detection with Music', cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_emotion = None

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            current_emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # If an emotion is detected and different from the last one
            if current_emotion != last_emotion and not music_playing:
                last_emotion = current_emotion
                emotion_start_time = time.time()  # Reset the timer
                timer_display = True  # Show the timer when a new emotion is detected
                timer_started = True  # Mark the timer as started

            # Check if the emotion is stable for the required duration
            if emotion_start_time is not None and timer_started:
                elapsed_time = time.time() - emotion_start_time

                # If the emotion is stable for the specified duration, play the music
                if elapsed_time >= emotion_stable_duration and not music_playing:
                    play_music(current_emotion)  # Start playing music for the detected emotion
                    timer_display = False  # Hide the timer after music starts
                    minimize_window_time = time.time() + 5  # Set the time to minimize the window

                # Display the elapsed time on the video frame if timer_display is True
                if timer_display:
                    cv2.putText(frame, f'Time: {int(elapsed_time)}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection with Music', frame)

    # Check if it's time to minimize the window
    if minimize_window_time and time.time() >= minimize_window_time:
        # Minimize the OpenCV window
        window = gw.getWindowsWithTitle('Real-time Emotion Detection with Music')[0]
        window.minimize()  # Minimize the window
        minimize_window_time = None  # Reset minimize window time

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        reset_emotion_detection()  # Reset emotions and music
        timer_display = True  # Start displaying timer again
        last_emotion = None  # Reset last emotion to capture a new one

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()  # Ensure we only close windows after the loop ends
