# Step 1: Base image with Python
FROM python:3.9-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install system dependencies (esp. for OpenCV and mediapipe)
# RUN apt-get update && apt-get install -y \
#     ffmpeg libsm6 libxext6 \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    gcc build-essential \
    default-libmysqlclient-dev pkg-config \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*


# Step 4: Copy project files
COPY . .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Run the app (change if it's a lib or API)
CMD ["python", "app.py"]
