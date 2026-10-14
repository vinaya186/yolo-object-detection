
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLO Object Detection")
st.write("Upload an image or video to detect objects using your trained YOLO model.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()

option = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Image Detection", "Video Detection"]
)

if option == "Image Detection":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                results = model.predict(image, verbose=False)
                result = results[0]
                annotated = cv2.cvtColor(
                    result.plot(),
                    cv2.COLOR_BGR2RGB
                )

            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "Object": model.names[class_id],
                    "Confidence": f"{confidence:.1%}"
                })

            st.write(f"**Total detections: {len(detections)}**")

            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No objects detected.")

            success, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            )

            if success:
                st.download_button(
                    "Download Result",
                    data=encoded.tobytes(),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

elif option == "Video Detection":
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file and st.button("Detect Objects in Video"):
        input_path = None
        output_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_input:
                temp_input.write(uploaded_file.getvalue())
                input_path = temp_input.name

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_output:
                output_path = temp_output.name

            with st.spinner("Processing video. Please wait..."):
                cap = cv2.VideoCapture(input_path)

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if not fps or fps <= 0:
                    fps = 25.0

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not cap.isOpened() or not writer.isOpened():
                    cap.release()
                    writer.release()
                    st.error("Could not open the video for processing.")
                    st.stop()

                total_detections = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = model.predict(frame, verbose=False)[0]
                    annotated = result.plot()
                    writer.write(annotated)
                    total_detections += len(result.boxes)

                cap.release()
                writer.release()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.success("Video detection completed!")
                st.write(f"Total object detections across frames: {total_detections}")

                with open(output_path, "rb") as video:
                    st.download_button(
                        "Download Detected Video",
                        data=video,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("No output video was generated.")

        except Exception as e:
            st.error(f"Video processing failed: {e}")

        finally:
            for path in [input_path, output_path]:
                if path and os.path.exists(path):
                    os.remove(path)
