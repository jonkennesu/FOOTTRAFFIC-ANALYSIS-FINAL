import streamlit as st
import cv2
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO, RTDETR
from PIL import Image
import os

# Streamlit page setup
st.set_page_config(page_title="Retail Detection System", layout="centered")
st.title("Retail Customer and Product Detection with Heatmap")

# Load models only once
@st.cache_resource
def load_models():
    people_model = YOLO("best.pt")
    product_model = RTDETR("weights.pt")
    return people_model, product_model

people_model, product_model = load_models()

# File uploader
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'last_uploaded_name' not in st.session_state:
    st.session_state.last_uploaded_name = None

# If file is uploaded
if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.processed = False
        st.session_state.last_uploaded_name = uploaded_file.name

    file_type = uploaded_file.type

    # IMAGE MODE
    if file_type.startswith("image"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Running Detection..."):
            people_results = people_model(img)
            product_results = product_model(img)

            st.subheader("Detected People")
            st.image(people_results[0].plot(), caption="People Detection", use_column_width=True)

            st.subheader("Detected Products")
            st.image(product_results[0].plot(), caption="Product Detection", use_column_width=True)

    # VIDEO MODE
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            temp_video_path = tmp.name

        if not st.session_state.processed:
            cap = cv2.VideoCapture(temp_video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps

            time_slice_sec = 10
            slice_frame_count = int(fps * time_slice_sec)
            num_slices = int(np.ceil(duration_sec / time_slice_sec))

            heatmaps = [np.zeros((frame_height, frame_width), dtype=np.float32) for _ in range(num_slices)]
            output_path = os.path.join(tempfile.gettempdir(), "combined_output.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            stframe = st.empty()
            progress = st.progress(0)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # People detection
                people_results = people_model.predict(source=img_rgb, conf=0.5, classes=[0], verbose=False)
                people_boxes = people_results[0].boxes

                # Update heatmap with people centers
                if people_boxes is not None:
                    for box in people_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= cy < frame_height and 0 <= cx < frame_width:
                            slice_index = frame_count // slice_frame_count
                            if slice_index < len(heatmaps):
                                cv2.circle(heatmaps[slice_index], (cx, cy), 10, 1, -1)

                # Product detection
                product_results = product_model(img_rgb)

                # Overlay results
                people_annot = people_results[0].plot()
                product_annot = product_results[0].plot()
                combined_annot = cv2.addWeighted(people_annot, 0.6, product_annot, 0.4, 0)

                out.write(cv2.cvtColor(combined_annot, cv2.COLOR_RGB2BGR))

                if frame_count % int(fps) == 0:
                    stframe.image(combined_annot, channels="RGB", use_container_width=True)

                frame_count += 1
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()

            st.session_state.heatmaps = heatmaps
            st.session_state.output_path = output_path
            st.session_state.processed = True

        st.success("Video processing complete.")

        st.subheader("Annotated Video with People and Product Detection")
        st.video(st.session_state.output_path)

        st.subheader("Select Heatmap Interval")
        interval_sec = 10
        intervals = [f"{i*interval_sec}s - {(i+1)*interval_sec}s" for i in range(len(st.session_state.heatmaps))]
        selected_slice = st.selectbox("Select interval:", intervals)
        selected_index = int(selected_slice.split('s')[0]) // interval_sec

        heat = st.session_state.heatmaps[selected_index]
        if heat.max() > 0:
            heat = heat / heat.max()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heat, cmap="Blues", ax=ax, cbar=True, xticklabels=False, yticklabels=False)
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"Foot Traffic Heatmap: {selected_slice}")
        st.pyplot(fig)

        st.download_button(
            label="Download Annotated Video",
            data=open(st.session_state.output_path, "rb").read(),
            file_name="combined_annotated_output.mp4",
            mime="video/mp4"
        )
