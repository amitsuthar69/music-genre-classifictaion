from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.utils import to_categorical
import uvicorn

templates = Jinja2Templates(directory="templates")

app = FastAPI()

model = tf.keras.models.load_model('./Music-genre-classification')
scaler = MinMaxScaler()

label_index = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

def predict_genre(model, scaler, audio_fp):
    # Load and process the external audio file
    audio_data, sr = librosa.load(audio_fp)
    audio_data, _ = librosa.effects.trim(audio_data)

    # Extract features from the audio file
    features = pd.DataFrame(columns=df.columns[1:])

    features.loc[0] = librosa.feature.zero_crossing_rate(y=audio_data)[0].mean(), \
                  librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0].mean(), \
                  librosa.feature.chroma_stft(y=audio_data, sr=sr).mean(), \
                  librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0].mean(), \
                  librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0].mean(), \
                  librosa.feature.mfcc(y=audio_data, sr=sr).mean(axis=1)

    # Normalize the features using the same scaler used during training
    normalized_features = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # Use the trained model to make predictions
    prediction = model.predict(normalized_features)

    # Map the predicted index to the genre label
    predicted_genre_index = np.argmax(prediction)
    predicted_genre = index_label[predicted_genre_index]

    return predicted_genre

@app.post("/predict")
async def predict_custom_audio_genre(file: UploadFile = File(...)):
    try:
        audio_fp = f"./temp/{file.filename}"
        with open(audio_fp, "wb") as buffer:
            buffer.write(file.file.read())

        # Use the predict_genre function to get the predicted genre
        predicted_genre = predict_genre(model, scaler, audio_fp)

        return JSONResponse(content={"predicted_genre": predicted_genre})
    except Exception as e:
        return JSONResponse(content={"error": f"Error during prediction: {str(e)}"}, status_code=500)


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)