FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# sounddevice 需要 PortAudio 运行库；curl 供 compose healthcheck 使用
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# P1：以 FastAPI 启动异步任务 API。
# 旧 Flask 入口仍然可用：python web/app.py（8080），迁移期两套并存。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
