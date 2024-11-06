#!/bin/bash
# Stage 1: Build stage
FROM  --platform=linux/amd64 python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git gcc g++ libgl1 libgl1-mesa-glx libglib2.0-0 poppler-utils tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app



# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
#COPY requirements.txt requirements.txt
#RUN pip install --user -r requirements.txt
# Install Poetry
RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache



COPY poetry.lock pyproject.toml ./
RUN touch README.md

# Install dependencies
RUN poetry install


# Stage 2: Runtime stage
FROM  --platform=linux/amd64 python:3.10-slim   as runtime

RUN apt-get update && apt-get install -y \
    gcc g++ libgl1 libgl1-mesa-glx libglib2.0-0 poppler-utils tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}



# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:/app/.venv/bin:$PATH

# Set environment variables from build args
ARG MIXTRAL_API_BASE
ARG MIXTRAL_API_KEY
ARG GPT_API_KEY
ARG OPENAI_API_URL

ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"

ENV MIXTRAL_API_BASE=$MIXTRAL_API_BASE
ENV MIXTRAL_API_KEY=$MIXTRAL_API_KEY
ENV GPT_API_KEY=$GPT_API_KEY
ENV OPENAI_API_URL=$OPENAI_API_URL

# Set working directory
WORKDIR /app

# Copy package code
COPY app app
COPY config.yml config.yml

# Expose port
EXPOSE 8501


# Set entrypoint
ENTRYPOINT ["/app/.venv/bin/python", "-m", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]