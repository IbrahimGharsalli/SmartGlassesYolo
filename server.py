import cvzone
from ultralytics import YOLO
import cv2
import math
from gtts import gTTS
import time

counter = 0
model = YOLO("../yolov8_Project/Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def run_program(source):
    cap = cv2.VideoCapture(source)

    detected_objects = {}  # Dictionnaire pour stocker le nombre d'occurrences de chaque objet détecté

    while True:
        success, img = cap.read()

        if not success:
            print("Erreur lors de la lecture de la source.")
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                detected_object = classNames[cls]  # Objet détecté

                # Mettre à jour le dictionnaire des objets détectés
                detected_objects[detected_object] = detected_objects.get(detected_object, 0) + 1

                cvzone.putTextRect(img, f'{detected_object} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convertir le dictionnaire des objets détectés en une chaîne de texte
    text_to_speak = " ".join([f'({count}) {obj} ' for obj, count in detected_objects.items()])

    # Convertir la liste des objets détectés en un fichier son
    start_time = time.time()
    tts = gTTS(text=text_to_speak, lang="en")
    tts.save("voice.wav")

    print("start")
    print("done")
    print("Elapsed time: {} seconds".format(time.time() - start_time))

# Spécifiez le chemin de la source ici (image ou vidéo)
source_path = "https://144f-196-177-23-41.ngrok-free.app/yolov5/captured_image.jpg"

# Définissez le délai entre chaque exécution en secondes (par exemple, 10 secondes)
delay_between_executions = 1

while True:
    run_program(source_path)
    time.sleep(delay_between_executions)
