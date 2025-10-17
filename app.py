import streamlit as st  # Bibliothèque d'interface web simple
import tempfile  # Pour stocker temporairement les fichiers uploadés
import os  # Pour la manipulation de chemins
import time  # Pour temporiser certaines actions

# Try to import OpenCV with error handling
try:
    import cv2
except ImportError:
    st.error("OpenCV could not be imported. Please check your installation.")
    st.stop()

# Try to import the detector with error handling
try:
    from detector import LeakDetector  # Importation de la classe de détection
except ImportError as e:
    st.error(f"Could not import LeakDetector: {e}")
    st.stop()

# Définir le thème Streamlit (large et sombre)
theme = "dark"
st.set_page_config(page_title="Water Leak Detection", layout="wide")

# Chemin du logo (dossier courant)
LOGO_PATH = os.path.join(os.path.dirname(__file__), "ocp-seeklogo.svg")

# --- Barre latérale ---
with st.sidebar:
    st.image(LOGO_PATH, width=120)  # Affiche le logo
    st.title("Paramètres de Détection")
    st.markdown("---")
    # L'utilisateur choisit le type d'entrée : image, vidéo, webcam
    input_type = st.radio("Sélectionner la source d'entrée :", ["Upload Image", "Upload Video", "Webcam"])
    st.markdown("---")
    st.caption("Plus d'options à venir ...")

# --- Zone principale ---
st.markdown(f"""
    <div style='display: flex; align-items: center;'>
        <img src="app/{os.path.basename(LOGO_PATH)}" width="60" style="margin-right: 20px;"/>
        <h1 style='margin-bottom: 0;'>💧 Détection automatique de fuite</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("""
Cette application utilise un modèle IA (YOLOv11) pour détecter automatiquement les fuites de produit à partir d'images, vidéos, ou de flux caméra en temps réel.

**Fonctionnalités :**
- Upload d'images ou de vidéos
- Surveillance en direct avec la webcam
- Télécharger les résultats annotés
""")

# Chargement du modèle
try:
    detector = LeakDetector(model_path=os.path.join(os.path.dirname(__file__), "best.pt"))
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# --- Détection sur IMAGE uploadée ---
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tfile.write(uploaded_file.read())  # Sauvegarde temporaire
        tfile.close()
        results, save_path = detector.detect_image(tfile.name, save_path=tfile.name + "_det.jpg")
        result_image = cv2.imread(save_path)
        # Affiche le résultat
        st.image(result_image, channels="BGR", caption="Résultat de la détection", use_column_width=True)
        # Propose de télécharger l'image annotée
        with open(save_path, "rb") as f:
            st.download_button("Télécharger l'image annotée", f, file_name="annotated.jpg")

# --- Détection sur VIDEO uploadée ---
elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("Importer une vidéo", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()
        cap = cv2.VideoCapture(video_path)
        # Récupère des infos sur la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        st.subheader("Informations sur la vidéo")
        st.write(f"Durée : {duration:.2f} secondes")
        st.write(f"Nombre de frames : {frame_count}")
        st.write(f"FPS : {fps}")
        # Affiche un slider pour choisir la frame
        st.subheader("Contrôles vidéo")
        progress_bar = st.progress(0)
        frame_slider = st.slider("Frame", 0, max(frame_count-1, 0), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
        ret, frame = cap.read()
        if ret:
            # Lance la prédiction sur la frame choisie
            results = detector.model(frame)
            annotated = detector._draw_boxes(frame.copy(), results)
            # Affiche l'image originale VS l'image annotée
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Frame originale")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                st.subheader("Frame traitée")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            progress_bar.progress(frame_slider / max(frame_count-1, 1))
        # Option : tout traiter
        if st.button("Traiter toute la vidéo"):
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
            frame_idx = 0
            with st.spinner("Traitement de la vidéo ..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = detector.model(frame)
                    annotated = detector._draw_boxes(frame.copy(), results)
                    out.write(annotated)
                    frame_idx += 1
                    progress_bar.progress(frame_idx / max(frame_count, 1))
                    if frame_idx % 10 == 0:
                        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Traitement frame {frame_idx}/{frame_count}", use_column_width=True)
                out.release()
            stats_placeholder.markdown("## Vidéo traitée avec succès !")
            # Propose de télécharger la vidéo annotée
            with open(processed_video_path, 'rb') as f:
                video_bytes = f.read()
                st.video(video_bytes)
            with open(processed_video_path, 'rb') as f:
                st.download_button(
                    label="Télécharger la vidéo traitée",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            os.unlink(processed_video_path)
        cap.release()
        os.unlink(video_path)

# --- Détection sur webcam ---
elif input_type == "Webcam":
    st.warning("Attention : l'accès à la webcam nécessite votre autorisation.")
    run_webcam = st.button("Démarrer la détection webcam")
    stop_webcam = st.button("Arrêter la détection webcam")
    webcam_placeholder = st.empty()
    if run_webcam:
        cap = cv2.VideoCapture(0)  # Utilise la webcam principale
        st.session_state["webcam_running"] = True
        while cap.isOpened() and st.session_state.get("webcam_running", True):
            ret, frame = cap.read()
            if not ret:
                st.error("Problème d'accès à la webcam !")
                break
            # Effectuer la détection
            results = detector.model(frame)
            annotated = detector._draw_boxes(frame.copy(), results)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
            # Arrêter si le bouton stop est activé
            if stop_webcam:
                st.session_state["webcam_running"] = False
                break
            time.sleep(0.01)
        cap.release()
        webcam_placeholder.empty()
