# Use a Python base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the entire application
COPY . .

# Expose the port Flask will listen on
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]