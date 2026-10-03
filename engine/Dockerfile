FROM python:3.12-slim

# yt-dlp needs a JS runtime to decode YouTube's nsig throttle signature; without
# one, media downloads fail with HTTP 403. deno is yt-dlp's default runtime and
# ships as a single static binary.
COPY --from=denoland/deno:bin-2.1.4 /deno /usr/local/bin/deno

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt yt-dlp

COPY . .

# Single worker: jobs live in process memory, so every request must hit the
# same process that started the job.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
