# Use slim Python 3.10 image for TensorFlow compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package setup and install the local src package
COPY setup.py .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy remaining project files
COPY config/ ./config/
COPY params.yaml .
COPY models/ ./models/
COPY app.py .
COPY main.py .

# Expose port — 7860 for Hugging Face Spaces, 8501 for local/other platforms
EXPOSE 7860

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
 
