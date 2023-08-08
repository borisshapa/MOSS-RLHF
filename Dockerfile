FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN set -x && \
    echo "Acquire { HTTP { Proxy \"$HTTP_PROXY\"; Pipeline-Depth 0; No-Cache true;}; BrokenProxy true;};" | tee /etc/apt/apt.conf

ARG INSTALL_DIR=/opt
ARG BUILD_DIR=/app

WORKDIR $BUILD_DIR

COPY requirements.txt $BUILD_DIR

ENV CUDA_PATH /usr/local/cuda
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git wget unzip libaio-dev libtinfo-dev

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN ln -sf $(which python3) /usr/bin/python && \
    ln -sf $(which pip3) /usr/bin/pip

ENV	LC_CTYPE en_US.UTF-8
ENV	LANG en_US.UTF-8

RUN pip install --upgrade pip
RUN pip install tensorboard datasets accelerate safetensors chardet cchardet
RUN pip install transformers sentencepiece einops rouge jionlp==1.4.14 nltk sacrebleu cpm_kernels wandb loguru
RUN git clone https://github.com/microsoft/DeepSpeed.git && \
    cd DeepSpeed && \
    pip3 install -e .
