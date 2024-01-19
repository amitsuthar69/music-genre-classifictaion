from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler # remove this if needed
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.utils import to_categorical
import uvicorn

scaler = StandardScaler()

templates = Jinja2Templates(directory="templates")
app = FastAPI()

model = tf.keras.models.load_model('./Music-genre-classification')

index_label = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

feature_names = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']

# Define feature extraction function
def extract_features(audio_data, sr):
    # Extract various audio features
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    rms = librosa.feature.rms(y=audio_data)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
    harmony, perceptr = librosa.effects.harmonic(audio_data), librosa.effects.percussive(audio_data)
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)

    # Combine features into a DataFrame
    external_data = pd.DataFrame({
        'chroma_stft_mean': np.mean(chroma_stft, axis=1),
        'chroma_stft_var': np.var(chroma_stft, axis=1),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_var': np.var(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_var': np.var(spectral_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'zero_crossing_rate_var': np.var(zero_crossing_rate),
        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),
        'perceptr_mean': np.mean(perceptr),
        'perceptr_var': np.var(perceptr),
        'tempo': tempo,
        'mfcc1_mean': np.mean(mfccs[0, :]),
        'mfcc1_var': np.var(mfccs[0, :]),
        'mfcc2_mean': np.mean(mfccs[1, :]),
        'mfcc2_var': np.var(mfccs[1, :]),
        'mfcc3_mean': np.mean(mfccs[2, :]),
        'mfcc3_var': np.var(mfccs[2, :]),
        'mfcc4_mean': np.mean(mfccs[3, :]),
        'mfcc4_var': np.var(mfccs[3, :]),
        'mfcc5_mean': np.mean(mfccs[4, :]),
        'mfcc5_var': np.var(mfccs[4, :]),
        'mfcc6_mean': np.mean(mfccs[5, :]),
        'mfcc6_var': np.var(mfccs[5, :]),
        'mfcc7_mean': np.mean(mfccs[6, :]),
        'mfcc7_var': np.var(mfccs[6, :]),
        'mfcc8_mean': np.mean(mfccs[7, :]),
        'mfcc8_var': np.var(mfccs[7, :]),
        'mfcc9_mean': np.mean(mfccs[8, :]),
        'mfcc9_var': np.var(mfccs[8, :]),
        'mfcc10_mean': np.mean(mfccs[9, :]),
        'mfcc10_var': np.var(mfccs[9, :]),
        'mfcc11_mean': np.mean(mfccs[10, :]),
        'mfcc11_var': np.var(mfccs[10, :]),
        'mfcc12_mean': np.mean(mfccs[11, :]),
        'mfcc12_var': np.var(mfccs[11, :]),
        'mfcc13_mean': np.mean(mfccs[12, :]),
        'mfcc13_var': np.var(mfccs[12, :]),
        'mfcc14_mean': np.mean(mfccs[13, :]),
        'mfcc14_var': np.var(mfccs[13, :]),
        'mfcc15_mean': np.mean(mfccs[14, :]),
        'mfcc15_var': np.var(mfccs[14, :]),
        'mfcc16_mean': np.mean(mfccs[15, :]),
        'mfcc16_var': np.var(mfccs[15, :]),
        'mfcc17_mean': np.mean(mfccs[16, :]),
        'mfcc17_var': np.var(mfccs[16, :]),
        'mfcc18_mean': np.mean(mfccs[17, :]),
        'mfcc18_var': np.var(mfccs[17, :]),
        'mfcc19_mean': np.mean(mfccs[18, :]),
        'mfcc19_var': np.var(mfccs[18, :]),
        'mfcc20_mean': np.mean(mfccs[19, :]),
        'mfcc20_var': np.var(mfccs[19, :]),
    })

    print(external_data)

    mean_values = external_data.mean()
    std_values = external_data.std()
   # Check for zero standard deviation
    if (std_values == 0).any():
        # Handle the case where standard deviation is zero (avoid division by zero)
        print("Warning: Zero standard deviation detected. Skipping normalization.")
        external_data_scaled = external_data.copy()
    else:
        # Normalize the external_data using calculated mean and std
        external_data_scaled = (external_data - mean_values) / std_values

    print(external_data_scaled)

    return external_data_scaled

# Define prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict_genre(request: Request, audio_file: UploadFile = File(...)):
    try:
        audio_data, sr = librosa.load(audio_file.file)
        features_scaled = extract_features(audio_data, sr)
        print("Features scaled:", features_scaled)
        prediction = model.predict(features_scaled)
        print(prediction)
        predicted_genre_index = np.argmax(prediction)%10
        res = (list(index_label.keys())[predicted_genre_index])
        #return res
        return templates.TemplateResponse(
        "index.html",
        {"request": request, "res": res},
    )
    except Exception as e:
        print(e)
        return str(e)

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)