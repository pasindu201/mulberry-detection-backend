# Use a smaller base image
FROM python:3.12.4-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# # Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 

# Copy the rest of the application code
COPY . /app/

# Expose port 8080 (same as in docker-compose)
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
