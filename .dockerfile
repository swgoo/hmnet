FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
WORKDIR /temp
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

RUN cd /temp
WORKDIR /workspace
RUN rm -rf /temp/*