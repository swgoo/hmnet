FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
WORKDIR /temp
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
WORKDIR /workspace
RUN rm -rf /temp/*