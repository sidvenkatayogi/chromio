FROM python:3.12

WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy the rest of the application
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run with Gunicorn for production
# Adjust workers (-w) based on your server's CPU cores
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--log-level", "info", "--access-logfile", "-", "api:create_app()"]