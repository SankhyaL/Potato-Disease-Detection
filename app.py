import streamlit as st
from PIL import Image
from utils import predict_cnn, predict_resnet, predict_yolo
import time
import pandas as pd

st.title("🥔 Potato Disease Detection App")
st.write("Upload an image of a potato leaf to detect disease using different ML models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

model_option = st.selectbox(
    "Select a model:",
    ("CNN", "ResNet", "YOLOv8")
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width='stretch')

    if st.button("Predict"):
        with st.spinner("Running model..."):
            if model_option == "CNN":
                result = predict_cnn.predict(uploaded_file)
            elif model_option == "ResNet":
                result = predict_resnet.predict(uploaded_file)
            else:
                result = predict_yolo.predict(uploaded_file)
        st.success(f"**Prediction:** {result}")

    # Summary button: runs all models, measures inference time and extracts confidence if available,
    # then shows a comparison bar chart for inference times (and confidences if present).
    if st.button("Summary"):
        with st.spinner("Generating summary..."):
            models = [
                ("CNN", predict_cnn),
                ("ResNet", predict_resnet),
                ("YOLOv8", predict_yolo),
            ]

            rows = []
            for name, mod in models:
                start = time.time()
                try:
                    res = mod.predict(uploaded_file)
                except Exception as e:
                    res = str(e)
                elapsed_ms = (time.time() - start) * 1000

                # try to extract label and confidence/score if the predict function returns them
                label = None
                confidence = None
                if isinstance(res, (list, tuple)):
                    if len(res) >= 1:
                        label = res[0]
                    if len(res) >= 2 and isinstance(res[1], (int, float)):
                        confidence = float(res[1])
                elif isinstance(res, dict):
                    label = res.get("label") or res.get("prediction") or res.get("pred") or str(res)
                    for key in ("confidence", "score", "prob", "probability"):
                        if key in res and isinstance(res[key], (int, float)):
                            confidence = float(res[key])
                            break
                else:
                    label = str(res)

                rows.append({
                    "Model": name,
                    "Label": label,
                    "Confidence": confidence,
                    "Time_ms": elapsed_ms
                })

            df = pd.DataFrame(rows).set_index("Model")
            st.write(df[["Label", "Confidence", "Time_ms"]])

            # show inference time comparison
            st.subheader("Inference time (ms) comparison")
            st.bar_chart(df["Time_ms"])

            # show confidence comparison if any confidences exist
            if df["Confidence"].notna().any():
                st.subheader("Confidence comparison")
                st.bar_chart(df["Confidence"])
