FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# for more information see:
# https://github.com/inb-uni-luebeck/docker/blob/master/example_workingdir/Dockerfile

# NOTE, build with: docker build -t dl_workingdir .

#https://code.visualstudio.com/docs/containers/quickstart-python

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo /home/user/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && chmod +x /home/user/miniconda.sh \
    && /home/user/miniconda.sh -b -p /home/user/miniconda \
    && rm /home/user/miniconda.sh \
    && conda install -y python==3.7 \
    && conda clean -ya

# Install PyTorch
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.4.0=py3.7_cuda10.0.130_cudnn7.6.3_0" \
    "torchvision=0.5.0=py37_cu100" \
    && conda clean -ya

# Install Jupyter Notebook
RUN conda install -y -c conda-forge jupyterlab

RUN pip install git+https://github.com/pgruening/dlbio
RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y

WORKDIR /other_py_packages
RUN git clone https://github.com/cybertronai/pytorch-lamb.git
RUN pip install -e ./pytorch-lamb

# for matplotlib
WORKDIR /.config/matplotlib
RUN chmod 777 /.config/matplotlib
WORKDIR /.cache
RUN chmod 777 /.cache

# selecting a work dir also creates the directory, so we use it to create /data as a mounting point
WORKDIR /data

# our actual working directory
WORKDIR /workingdir

#make port 8888 available from outside, just in case the container is supposed to run jupyter notebooks
EXPOSE 8888

# add application ressources to container
ADD . /workingdir

# run app
#CMD ["python", "datasets/ds_natural_images.py"]
#CMD ["python", "run_training.py"]