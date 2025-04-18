# analyzer/main.py

import os
import tempfile
import requests
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ─── Configurazione Supabase ────────────────────────────────────────────────

SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service Role Key
if not SUPA_URL or not SUPA_KEY:
    raise RuntimeError("Le variabili SUPABASE_URL e SUPABASE_SERVICE_KEY devono essere impostate")

supabase: Client = create_client(SUPA_URL, SUPA_KEY)

# ─── FastAPI App e CORS ─────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # o ["*"] per aprire a tutti i client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modello di richiesta ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    track_id: str

# ─── Endpoint di analisi ─────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # 1. Recupera il record track
    resp = supabase.table("tracks").select("*").eq("id", req.track_id).single().execute()
    if resp.error or not resp.data:
        raise HTTPException(status_code=404, detail="Track non trovata")
    track = resp.data
    file_url = track["file_url"]

    # 2. Scarica il file in un file temporaneo
    suffix = os.path.splitext(file_url)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    r = requests.get(file_url)
    try:
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Errore nel download del file: {e}")
    tmp.write(r.content)
    tmp.flush()

    # 3. Carica l’audio con librosa
    try:
        y, sr = librosa.load(tmp.name, sr=None, mono=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel caricamento audio: {e}")

    # 4. Calcoli base
    # 4.1 RMS loudness
    rms = float(librosa.feature.rms(y=y if y.ndim == 1 else y.mean(axis=0)).mean())
    # 4.2 Peak level
    peak = float(max(abs(y.flatten())))
    # 4.3 Stereo width (spectral contrast mean as proxy)
    spect_contrast = librosa.feature.spectral_contrast(y=y if y.ndim == 1 else y.mean(axis=0), sr=sr)
    stereo_width = float(spect_contrast.mean())

    # 4.4 Frequencies balance (low/mid/high bins)
    S = abs(librosa.stft(y if y.ndim == 1 else y.mean(axis=0)))
    freqs = librosa.fft_frequencies(sr=sr)
    # define bands
    low_idx = freqs < 200
    mid_idx = (freqs >= 200) & (freqs < 2000)
    high_idx = freqs >= 2000
    balance = {
        "low": float(S[low_idx].mean()),
        "mid": float(S[mid_idx].mean()),
        "high": float(S[high_idx].mean())
    }

    # 5. Feedback testuale semplice
    notes = []
    if rms < 0.01:
        notes.append("Traccia troppo silenziosa: alza il volume.")
    if peak > 0.9:
        notes.append("Attenzione al clipping: abbassa i picchi.")
    if stereo_width < 10:
        notes.append("Campo stereo stretto: considera di ampliare la larghezza.")
    if not notes:
        notes.append("Analisi completata: mix in media ok.")

    fb = {
        "track_id": req.track_id,
        "loudness_db": rms,
        "peak_db": peak,
        "stereo_width": stereo_width,
        "freq_balance": balance,
        "ai_notes": " ".join(notes),
    }

    # 6. Scrivi il feedback in Supabase
    insert_resp = supabase.table("feedback").insert(fb).execute()
    if insert_resp.error:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio feedback: {insert_resp.error.message}")

    return {"status": "analyzed", "feedback": fb}
