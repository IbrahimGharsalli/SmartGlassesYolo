import cv2
import time
import os
import requests
import pygame
from pygame.locals import *

def run_program():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        if not success:
            print("Erreur lors de la lecture de la caméra.")
            break

        cv2.imshow("Webcam", img)

        # Attend que la touche 'c' soit pressée pour capturer et enregistrer l'image
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Enregistrez l'image avec le même nom de fichier à chaque capture
            image_filename = "captured_image.jpg"
            image_path = os.path.join("/home/cyber/Desktop/test/yolo/yolov5", image_filename)
            cv2.imwrite(image_path, img)

            print("Image mise à jour:", image_path)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def play_audio_from_url(url):
    # Initialisez Pygame et le module mixer
    pygame.init()
    pygame.mixer.init()

    # Fetch the audio file from the URL
    response = requests.get(url)

    if response.status_code == 200:
        # Enregistrez le contenu audio dans un fichier temporaire
        audio_temp_filename = "temp_audio.wav"
        with open(audio_temp_filename, "wb") as f:
            f.write(response.content)

        # Chargez le fichier audio avec Pygame
        sound = pygame.mixer.Sound(audio_temp_filename)

        # Jouez le fichier audio
        sound.play()

        # Attendez que la lecture de l'audio soit terminée
        pygame.time.wait(int(sound.get_length() * 1000))

        # Supprimez le fichier audio temporaire
        os.remove(audio_temp_filename)

    else:
        print(f"Failed to fetch audio from URL. Status code: {response.status_code}")

    # Quittez Pygame
    pygame.mixer.quit()
    pygame.quit()

# Example URL (replace with your actual URL)
audio_url = "https://d06d-196-235-85-173.ngrok-free.app/voice.wav"

# Run the webcam program
run_program()
time.sleep(4)
# Jouez l'audio
play_audio_from_url(audio_url)

