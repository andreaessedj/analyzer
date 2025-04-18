import os
import tempfile
import requests
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

# Configurazione
SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service Role Key
supabase: Client = create_client(SUPA_URL, SUPA_KEY)

app = FastAPI()

class AnalyzeRequest(BaseModel):
    track_id: str

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # 1. Prendi il record della traccia
    resp = supabase.table("tracks").select("*").eq("id", req.track_id).single().execute()
    if resp.error or not resp.data:
        raise HTTPException(404, "Track non trovata")
    file_url = resp.data["file_url"]

    # 2. Scarica il file in /tmp
    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_url)[1], delete=False)
    r = requests.get(file_url)
    r.raise_for_status()
    tmp.write(r.content)
    tmp.flush()

    # 3. Carica con librosa
    y, sr = librosa.load(tmp.name, sr=None, mono=False)
    rms = float(librosa.feature.rms(y=y).mean())
    peak = float(max(abs(y.flatten())))
    # Spettro semplificato
    S = librosa.stft(y.mean(axis=0) if y.ndim>1 else y)
    freqs = librosa.fft_frequencies(sr=sr)
    # Qui potresti dividere in low/mid/high…

    # 4. Scrivi il feedback
    fb = {
      "track_id": req.track_id,
      "loudness_db": rms,
      "peak_db": peak,
      "stereo_width": float(librosa.feature.spectral_contrast(y=y, sr=sr).mean()),
      "freq_balance": {"low":0.3,"mid":0.4,"high":0.3},  # esempio
      "ai_notes": "Mix ok, ma i bassi sono un po' alti",
    }
    supabase.table("feedback").insert(fb).execute()

    return {"status": "analyzed", "feedback": fb}
