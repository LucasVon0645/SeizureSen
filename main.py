import streamlit as st
import os

from src.FrontEnd.app import app as frontend_app
from src.SeizureSenPredictor.SeizureSenPredictor import SeizureSenPredictor

st.set_page_config(
    page_title="EEG Seizure Prediction", page_icon=":dog:", layout="wide"
)


@st.cache_resource
def load_model():
    model_config_path = os.path.join(
        "models",
        "best_model",
        "model_config.json",
    )
    return SeizureSenPredictor(model_config_path)


seizure_sen_predictor = load_model()
frontend_app(seizure_sen_predictor)
