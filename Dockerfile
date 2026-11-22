FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_HOME=/opt/torch
WORKDIR /app

RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# Bake the segmentation weights into the image so the container never downloads at runtime
RUN python -c "from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50; deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)"

COPY scripts scripts
COPY web web
COPY models models

RUN mkdir -p /app/submissions && useradd -m app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "server:app", "--app-dir", "scripts", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
