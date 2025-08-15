# Hugging Face Space (Docker) for H2O + Streamlit
FROM python:3.11-slim

# Install Java runtime for H2O
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set workdir
WORKDIR /app

# Copy requirements first and install (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py ./
COPY train_h2o.py ./
COPY models ./models

# Streamlit will use this port on Spaces
ENV PORT=7860
ENV H2O_MAX_MEM=2G

# Expose port (optional for Spaces but helpful for local Docker run)
EXPOSE 7860

# Default command for Spaces
CMD ["bash", "-lc", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
