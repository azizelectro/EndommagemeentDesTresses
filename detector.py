try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import streamlit as st
    st.error("OpenCV could not be imported. Please check your installation.")
    st.stop()

from ultralytics import YOLO  # Importation du modèle YOLO pour la détection IA
import os  # Module pour gérer les chemins de fichiers
from typing import Optional, Tuple, List  # Types Python pour la clarté

class LeakDetector:
    def __init__(self, model_path: str = "updated_model.pt"):
        """
        Initialise le détecteur avec un modèle pré-entraîné.
        model_path : chemin du modèle YOLOv11.
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is not available. Cannot initialize LeakDetector.")
        
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # Noms des classes du modèle (non utilisé ici)

    def detect_image(self, image_path: str, save_path: Optional[str] = None) -> Tuple[List, str]:
        """
        Prend une image en entrée et y détecte les fuites.
        image_path : chemin de l'image.
        save_path : où sauvegarder la version annotée (optionnel).
        Retourne les résultats et le chemin de sauvegarde.
        """
        image = cv2.imread(image_path)
        results = self.model(image)  # Exécute la détection
        annotated = self._draw_boxes(image, results)  # Dessine les boîtes
        if save_path:
            cv2.imwrite(save_path, annotated)  # Sauvegarde l'image annotée
        return results, save_path if save_path else ""

    def detect_video(self, video_path: str, save_path: Optional[str] = None, notify_callback=None):
        """
        Analyse une vidéo pour détecter les fuites image par image.
        video_path : chemin vidéo
        save_path : vidéo annotée à créer (optionnel)
        notify_callback : fonction à appeler lorsqu'une fuite est détectée (optionnel)
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fin de la vidéo
            results = self.model(frame)
            annotated = self._draw_boxes(frame, results)
            if out:
                out.write(annotated)
            if notify_callback:
                notify_callback(results, annotated)  # Notification optionnelle
        cap.release()
        if out:
            out.release()

    def detect_stream(self, stream_url: str, save_path: Optional[str] = None, notify_callback=None):
        """
        Pour analyser un flux vidéo (ex: caméra IP) en temps réel.
        stream_url : adresse du flux
        save_path : fichier vidéo annoté (optionnel)
        notify_callback : fonction notification (optionnel)
        """
        cap = cv2.VideoCapture(stream_url)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_path:
            fps = 20  # Par défaut pour les streams
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated = self._draw_boxes(frame, results)
            if out:
                out.write(annotated)
            if notify_callback:
                notify_callback(results, annotated)
            cv2.imshow("Leak Detection", annotated)  # Affiche en temps réel
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Appuyez sur q pour quitter
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def _draw_boxes(self, frame, results):
        """
        Dessine les boîtes et annotations sur une image/vidéo
        frame : image à modifier
        results : résultats YOLOv11
        Retourne l'image annotée.
        """
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.int().cpu().tolist()  # Coordonnées des boîtes
                class_ids = result.boxes.cls.int().cpu().tolist()  # IDs des classes
                for box, class_id in zip(boxes, class_ids):
                    x1, y1, x2, y2 = box  # Coordonnées coin haut gauche et bas droit
                    label = "endommagement des tresses"  # Texte à afficher systématiquement
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rectangle rouge
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Texte vert
        return frame
