FROM hub.hamdocker.ir/python:3.12

# Install system dependencies needed for some packages
RUN apt-get update && apt-get install -y \
    libevent-dev \
    python3-psycopg2 \
    python3-dev \
    libpq-dev \
    build-essential \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt ./

# Install Python dependencies using wheels if available
RUN pip install --only-binary=:all: -r requirements.txt

# Copy the rest of the app
COPY ./ ./

# Default command
CMD python manage.py migrate && python manage.py runserver --noreload 0.0.0.0:8000
