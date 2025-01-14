import streamlit as st

from src.FrontEnd.app import app as frontend_app
from src.SeizureSenPredictor.SeizureSenPredictor import SeizureSenPredictor

st.set_page_config(
        page_title="EEG Seizure Prediction", page_icon=":dog:", layout="wide"
)

@st.cache_resource
def load_model():
    return SeizureSenPredictor("models/config/model_with_attention_10_steps_cfg.json")

seizure_sen_predictor = load_model()
frontend_app(seizure_sen_predictor)
