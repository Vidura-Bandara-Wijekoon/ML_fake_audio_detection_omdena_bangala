import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/'

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import librosa
import soundfile as sf
import xgboost as xgb
import uvicorn
import logging
import pickle
import io
from pydub import AudioSegment
from typing import List
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    logger.info("Serving the index page")
    with open("templates/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("header.webm", 'rb') as source_file:
    header_data = source_file.read(1024)

is_detecting = False

model = xgb.XGBClassifier()
model.load_model("models/xgboost.json")

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

q = deque()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)

manager = ConnectionManager()

def extract_features(audio, sr=16000):

    target_duration = 5
    target_samples = int(target_duration * sr)

    # Get the current number of samples
    current_samples = len(audio)

    # If the audio is shorter than 5 seconds, pad with zeros
    if current_samples < target_samples:
        padding_samples = target_samples - current_samples
        audio = np.concatenate([audio, np.zeros(padding_samples)])
    else:
        audio = audio[:target_samples]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).flatten()
    mfcc = np.array(mfcc)

    return mfcc

async def process_audio_data(audio_data):
    try:  
        full_audio_data = header_data + audio_data

        audio_segment = AudioSegment.from_file(io.BytesIO(full_audio_data), format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        audio, sr = sf.read(wav_io, dtype='float32')
    except Exception as e:
        logger.error(f"Failed to read audio data: {e}")
        return

    if audio.ndim > 1:  # If audio has more than one channel, average them
        audio = np.mean(audio, axis=1)
    
    features = extract_features(audio)
    features = scaler.transform(features.reshape(1, -1))
    # features = features.reshape(1, -1)
    prediction = model.predict(features)
    is_fake = prediction[0]
    result = 'fake' if is_fake else 'real'
    q.append(is_fake)
    if len(q) > 2:
        if sum(q) == 2:
            for connection in manager.active_connections:
                await manager.send_message(connection, "global-fake")
            q.clear()
        else:
            q.popleft()
    
    return result

@app.post("/start_detection")
async def start_detection():
    global is_detecting

    if not is_detecting:
        is_detecting = True
    return JSONResponse(content={'status': 'detection_started'})

@app.post("/stop_detection")
async def stop_detection():
    global is_detecting
    is_detecting = False
    return JSONResponse(content={'status': 'detection_stopped'})

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:

        audio_data = await file.read()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=file.filename.split('.')[-1])
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        audio, sr = sf.read(wav_io, dtype='float32')

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        features = extract_features(audio)
        features = scaler.transform(features.reshape(1, -1))
        
        prediction = model.predict(features)
        is_fake = prediction[0]
        result = 'fake' if is_fake else 'real'
        
        return JSONResponse(content={'status': 'success', 'result': result})
    
    except Exception as e:
        logger.error(f"Failed to process audio file: {e}")
        return JSONResponse(content={'status': 'error', 'message': str(e)}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            result = await process_audio_data(data)
            if result:
                await manager.send_message(websocket, result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=7860)
