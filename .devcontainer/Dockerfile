FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USERNAME=roy
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HISTFILE=/home/roy/commandhistory/.bash_history \
    HISTSIZE=10000 \
    HISTFILESIZE=20000

RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

RUN printf '\nexport PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND; }history -a; history -n"\n' >> /home/roy/.bashrc \
    && printf '"\\e[A": history-search-backward\n"\\e[B": history-search-forward\n' >> /home/roy/.inputrc \
    && echo 'source /workspaces/gnn_explorer/.venv/bin/activate' >> /home/roy/.bashrc \
    && chown -R roy:roy /home/roy

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    openssh-client \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y tmux vim

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -U bitsandbytes>=0.46.1
RUN pip install torch torchvision torchaudio
RUN pip install torch_geometric

RUN mkdir -p /home/roy/commandhistory && chown -R roy:roy /home/roy/commandhistory

RUN echo 'export PROMPT_COMMAND="history -a; history -n"' >> /home/roy/.bashrc 

RUN echo 'if [ -f /workspaces/gnn_explorer/.venv/bin/activate ]; then source /workspaces/gnn_explorer/.venv/bin/activate; fi' >> /home/roy/.bashrc
