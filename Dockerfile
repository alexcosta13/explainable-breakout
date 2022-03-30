FROM tensorflow/tensorflow:latest-gpu

WORKDIR /explainable-breakout

# Define environment variables
ARG WANDB_SECRET=""

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install atari gym
RUN conda update -n base -c defaults conda
RUN conda config --set channel_priority strict
RUN conda install -c conda-forge gym-atari

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install other dependencies
RUN pip install tensorflow-gpu && \
    pip install wandb && \
    pip install pyyaml
RUN wandb login $WANDB_SECRET

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install opencv-python

# Copy source code
COPY src ./src
WORKDIR /explainable-breakout/src

# We need to define the command to launch when we are going to run the image.
# We use the keyword 'CMD' to do that.
# The following command will execute "python ./main.py".
CMD [ "python", "./main.py" ]