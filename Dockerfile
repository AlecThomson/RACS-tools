FROM continuumio/miniconda3

RUN echo "Updating apt repositories"
RUN apt update && apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y \
build-essential \
gfortran
RUN apt autoremove -y
RUN apt clean -y
# RUN cd / \
# && git clone https://github.com/AlecThomson/RACS-tools
RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache
WORKDIR ./
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ADD . /tmp/
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate racs-tools" > ~/.bashrc
ENV PATH /opt/conda/envs/racs-tools/bin:$PATH
