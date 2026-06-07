# AI Video Dubbing Backend — English to Hindi

A backend-only, self-hosted FastAPI application that:

1. accepts MP4, MOV, AVI, and MKV uploads;
2. extracts English audio with FFmpeg;
3. transcribes it locally with Faster-Whisper;
4. translates English segments to Hindi with MarianMT;
5. synthesizes Hindi speech locally with MMS/VITS;
6. time-fits every generated segment to the original timestamps;
7. creates a complete Hindi audio timeline; and
8. replaces the original audio and produces a downloadable MP4.

No frontend is included.

## Important architecture decision

Edge-TTS is intentionally **not used**. The Python `edge-tts` package sends text to Microsoft's online Edge speech service. That conflicts with a strict self-hosted/no-cloud requirement. This backend uses the locally executed `facebook/mms-tts-hin` VITS model instead.

The models are downloaded from their model repositories on first use. After they are cached, set `OFFLINE_MODE=true` to prevent network model access.

## Stack

- FastAPI and Uvicorn
- SQLite
- FFmpeg and ffprobe
- Faster-Whisper
- MarianMT: `Helsinki-NLP/opus-mt-en-hi`
- MMS/VITS Hindi TTS: `facebook/mms-tts-hin`
- PyTorch, Transformers, NumPy, and SoundFile
- A local thread worker; Redis/Celery are not required

## Requirements

- Python 3.11 recommended
- FFmpeg with ffprobe available on `PATH`
- At least 8 GB RAM for smaller Whisper models
- Considerably more RAM or a CUDA GPU for `large-v3`
- Disk space for model caches and uploaded/generated videos

`large-v3` is accurate but heavy. For normal CPU development, begin with:

```env
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

## Local installation

### 1. Install FFmpeg

Windows:

```powershell
winget install Gyan.FFmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1
```

macOS:

```bash
brew install ffmpeg libsndfile
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

### 2. Create the Python environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install the PyTorch build matching the installed CUDA version by following the official PyTorch installation command, then install the remaining requirements.

### 3. Configure the backend

```bash
copy .env.example .env
```

On Linux/macOS:

```bash
cp .env.example .env
```

Change settings in `.env` as needed.

### 4. Download models in advance

```bash
python -m scripts.download_models
```

This is optional. The first processing job also downloads missing models.

After downloading everything, strict offline execution can be enabled:

```env
OFFLINE_MODE=true
```

### 5. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- API documentation: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`

## Docker

```bash
cp .env.example .env
docker compose up --build
```

For an NVIDIA Container Toolkit environment:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The `data` and `models` directories are mounted so jobs and models survive container recreation.

## API

### Upload

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.mp4"
```

Response:

```json
{
  "job_id": "d6cc41d7140f4eddb84cc50e937a15ad",
  "status": "queued",
  "status_url": "http://localhost:8000/api/status/d6cc41d7140f4eddb84cc50e937a15ad"
}
```

### Status

```bash
curl "http://localhost:8000/api/status/JOB_ID"
```

### Job history

```bash
curl "http://localhost:8000/api/jobs?limit=20&offset=0"
```

### Logs

```bash
curl "http://localhost:8000/api/jobs/JOB_ID/logs"
```

### Transcript

```bash
curl "http://localhost:8000/api/jobs/JOB_ID/transcript"
```

### Hindi translation

```bash
curl "http://localhost:8000/api/jobs/JOB_ID/translation"
```

### Preview

```text
GET /api/jobs/{job_id}/preview
```

### Download

```bash
curl -L "http://localhost:8000/api/download/JOB_ID" \
  --output dubbed_hindi.mp4
```

### Delete a completed or failed job

```bash
curl -X DELETE "http://localhost:8000/api/jobs/JOB_ID"
```

## Progress stages

- 0–5%: queued and media validation
- 10%: audio extraction
- 20–44%: transcription/model loading
- 45–59%: translation
- 60–85%: Hindi TTS and segment fitting
- 92%: video/audio merge
- 98%: output verification
- 100%: complete

Model download time occurs inside the first relevant stage and may make that percentage appear stationary.

## Synchronization method

Each translated segment keeps the source Whisper start and end times. The backend:

1. synthesizes the Hindi segment;
2. calculates the required duration;
3. uses chained FFmpeg `atempo` filters to fit the speech;
4. pads or trims to the exact segment window;
5. places the segment at the matching timestamp in a disk-backed NumPy audio timeline; and
6. normalizes overlapping speech to prevent clipping.

This is timeline synchronization, not true visual lip synchronization. Accurate lip movement requires a separate lip-sync model and video re-rendering.

## Audio behavior

The final output replaces the source audio completely. It does not preserve music or sound effects separately, since that requires vocal/source separation such as Demucs. Adding source separation is possible, but it increases processing time and hardware requirements.

## Storage

Generated data is placed under:

```text
data/
├── jobs.sqlite3
├── uploads/
├── outputs/
├── work/
├── transcripts/
├── translations/
└── logs/
```

Model caches are placed under:

```text
models/
├── whisper/
└── huggingface/
```

## Cleanup

Delete old completed/failed jobs using:

```bash
python -m scripts.cleanup_old_jobs --days 7
```

## Production notes

- Keep `WORKER_COUNT=1` when one GPU is used. Multiple workers may load duplicate large models and exhaust VRAM.
- Put the API behind Nginx or Caddy for TLS and request-size control.
- Set a specific `ALLOWED_ORIGINS` list instead of `*`.
- Store data on a dedicated volume.
- Add authentication before exposing upload/download endpoints publicly.
- For multi-server deployment, replace the local thread worker and SQLite with Celery/RQ plus Redis and PostgreSQL.
- Check the licences of all selected models before commercial deployment. Free access does not automatically mean unrestricted commercial use.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
