import streamlit as st
import tempfile
import os
import time

# Try to import OpenCV with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("OpenCV could not be imported. Please check your installation.")
    st.stop()

# Try to import the detector with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("YOLO could not be imported. Please check your installation.")
    st.stop()

# Set page config
st.set_page_config(page_title="Water Leak Detection", layout="wide")

# Simple interface
st.title("💧 Water Leak Detection App")

# Check if model file exists
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it's in the repository.")
    st.stop()

# Load model
try:
    model = YOLO(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Simple image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    try:
        # Read image
        image = cv2.imread(tfile.name)
        if image is None:
            st.error("Could not read the uploaded image.")
        else:
            # Run detection
            results = model(image)
            
            # Draw results
            annotated = image.copy()
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.int().cpu().tolist()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, "Leak", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                
    except Exception as e:
        st.error(f"Error processing image: {e}")
    finally:
        # Clean up
        os.unlink(tfile.name)
