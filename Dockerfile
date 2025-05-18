# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and app code
COPY bert_sentiment ./bert_sentiment
COPY app.py .

# Expose the port Flask will run on
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]