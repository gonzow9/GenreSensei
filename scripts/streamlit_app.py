"""
Streamlit demo. Run with:
    streamlit run scripts/streamlit_app.py
Requires: pip install streamlit
"""

from pathlib import Path
import sys

# Ensure local package is importable when running without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  
from genre_classifier.config import load_config
from genre_classifier.predict import predict_audio


def main() -> None:
    st.title("GTZAN Genre Classifier")
    st.write("Upload an audio file and get SVM ensemble predictions.")

    cfg = load_config(None)
    uploaded = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav", "mp3"])
    if not uploaded:
        return

    tmp_path = Path("artifacts/uploads") / uploaded.name
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(uploaded.read())

    with st.spinner("Predicting..."):
        results = predict_audio(tmp_path, cfg)

    st.subheader("Predictions")
    for model_name, (genre, confidence) in results.items():
        st.write(f"{model_name.upper()}: **{genre}** ({confidence:.2f} confidence)")


if __name__ == "__main__":
    main()
