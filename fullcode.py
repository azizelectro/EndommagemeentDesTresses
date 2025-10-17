import cv2  # Pour le traitement d'images et vidéos
from ultralytics import YOLO  # Pour charger et utiliser le modèle IA
import cvzone  # Pour afficher du texte facilement sur l'image

# Chargement du modèle YOLOv11 pré-entraîné
model = YOLO("best.pt")
names = model.names  # Liste des classes

# Ouvrir un flux vidéo : peut être un fichier vidéo ou une webcam selon le cas d'utilisation
cap = cv2.VideoCapture('')  # Mettre un chemin vers une vidéo ou 0 pour webcam

while True:
    # Lire une image/une frame
    ret, frame = cap.read()
    if not ret:
        break  # On arrête s'il n'y a plus d'image

    frame = cv2.resize(frame, (640, 500))  # Redimensionne l'image pour faciliter le traitement

    # Appliquer la détection YOLO
    results = model.track(frame, persist=True)

    # Vérifier s'il y a des détections (boîtes)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Récupérer boîtes, IDs, types, scores de confiance
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Coordonnées
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # ID de la classe
        track_ids = results[0].boxes.id.int().cpu().tolist()  # ID de suivi/track de chaque objet
        confidences = results[0].boxes.conf.cpu().tolist()  # Confiance du modèle

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]  # Nom de la classe détectée
            x1,y1,x2,y2=box  # Coordonnées du rectangle
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)  # Trace le rectangle rouge
            cvzone.putTextRect(frame, f'{c}', (x1,y1), 1, 1)  # Affiche le nom de la classe

    # Afficher le résultat dans une fenêtre
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Quitter en appuyant sur "q"

# Libérer les ressources et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
