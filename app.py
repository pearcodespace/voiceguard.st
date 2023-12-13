import streamlit as st

import numpy as np
import pandas as pd
import librosa
import joblib

model = joblib.load('random_forest_model.joblib')

st.title('REAL vs FAKE: ')

with st.sidebar:
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML"])
    
if choice == "Upload":
    st.title("Upload Your File for Detecting!")
    file = st.file_uploader("Upload Your Audio Here")
    if file:
        # Define a function to extract audio features
        def extract_audio_features(file):
            y, sr = librosa.load(file, duration = 0.1)
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)

            return chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfccs
        
        # Extract audio features from the input file
        chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfccs = extract_audio_features(file)
        # Organize the features into a DataFrame with the same structure as your training data
        input_data = pd.DataFrame({
            'chroma_stft': [chroma_stft],
            'rms': [rms],
            'spectral_centroid': [spectral_centroid],
            'spectral_bandwidth': [spectral_bandwidth],
            'rolloff': [rolloff],
            'zero_crossing_rate': [zero_crossing_rate],
            'mfcc1': [mfccs[0]],
            'mfcc2': [mfccs[1]],
            'mfcc3': [mfccs[2]],
            'mfcc4': [mfccs[3]],
            'mfcc5': [mfccs[4]],
            'mfcc6': [mfccs[5]],
            'mfcc7': [mfccs[6]],
            'mfcc8': [mfccs[7]],
            'mfcc9': [mfccs[8]],
            'mfcc10': [mfccs[9]],
            'mfcc11': [mfccs[10]],
            'mfcc12': [mfccs[11]],
            'mfcc13': [mfccs[12]],
            'mfcc14': [mfccs[13]],
            'mfcc15': [mfccs[14]],
            'mfcc16': [mfccs[15]],
            'mfcc17': [mfccs[16]],
            'mfcc18': [mfccs[17]],
            'mfcc19': [mfccs[18]],
            'mfcc20': [mfccs[19]]
        })
        input_data
        
        def predict():
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("The audio is predicted as FAKE :thumbsdown:")
            else:
                st.success("The audio is predicted as REAL:thumbsup:")
        st.button('Predict', on_click=predict)