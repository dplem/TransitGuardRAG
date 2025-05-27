FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install CPU-only torch first to avoid CUDA/CUDNN hash issues
RUN pip install --upgrade pip && pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "start.sh"] 