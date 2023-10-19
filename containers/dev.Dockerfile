FROM python:3.10-slim

ENV NVIDIA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

WORKDIR /tctsti
COPY . .

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    graphviz \
    graphviz-dev

RUN pip install --upgrade pip && \
    pip install -e .[dev]

LABEL org.opencontainers.image.title="tctsti" \
      org.opencontainers.image.authors="tctsti team" \
      org.opencontainers.image.description="This image contains the tctsti library." \
      org.opencontainers.image.url="https://github.com/pyrovelocity/tctsti" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
