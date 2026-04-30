import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print("Loading data and models...")

df = pd.read_csv("data/cleaned_dataset.csv")

df['year'] = (df['date_id'] // 365) + 2000
df['month'] = ((df['date_id'] % 365) // 30) + 1
df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(df['date_id'], unit='D')

try:
    with open("models/best_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model_loaded = True
except:
    model_loaded = False

try:
    with open("analysis_summary.json", "r") as f:
        summary = json.load(f)
except:
    summary = {}

print("✓ Data and models loaded")
