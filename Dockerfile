FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    BIORAGENT_DATA_DIR=/app/data

WORKDIR /app

COPY app/ .

RUN groupadd --system --gid 10001 bioragent \
    && useradd --system --uid 10001 --gid bioragent --home-dir /tmp --no-create-home bioragent \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /app/data \
    && chown -R bioragent:bioragent /app/data

EXPOSE 7860

USER bioragent:bioragent

CMD ["python", "app.py"]
