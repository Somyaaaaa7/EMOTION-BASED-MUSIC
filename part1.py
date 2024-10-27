import cv2
import random
import pygame
from deepface import DeepFace

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

# Initialize a variable to hold the last detected emotion
last_emotion = None
music_playing = None

def play_music(emotion):
    global music_playing
    # Stop the currently playing music if it exists
    if music_playing:
        pygame.mixer.music.stop()
    
    # Ensure the emotion is valid and available in the music dictionary
    if emotion in emotion_to_music:
        # Choose a random song for the detected emotion
        music_file = random.choice(emotion_to_music[emotion])
        
        # Load and play the new music
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(-1)  # -1 means the music will loop indefinitely
        music_playing = music_file
    else:
        print(f"No music available for emotion: {emotion}")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Update last_emotion for music selection
            if emotion != last_emotion:
                last_emotion = emotion
                play_music(emotion)

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        # Default to neutral if no face is detected
        last_emotion = "neutral"  # You can choose not to play any music for this case

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection with Music', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
