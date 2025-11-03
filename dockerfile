FROM python:3.11-slim

# Install system dependencies for OpenCV and RAW processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libraw-dev \
    dcraw \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Create necessary directories
RUN mkdir -p /app/uploads /app/processed

EXPOSE 3683

CMD ["python", "app.py"]