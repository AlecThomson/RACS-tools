FROM mambaorg/micromamba
USER root
RUN echo "Updating apt repositories"
RUN apt update && apt upgrade -y && apt install -y git
USER $MAMBA_USER
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /src
RUN mkdir /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
COPY --chown=$MAMBA_USER:$MAMBA_USER . ./src
RUN echo "Installing python and uv"
RUN micromamba install python=3.8 uv -y -c conda-forge && \
    micromamba clean --all --yes
RUN echo "Installing RACS-tools"
RUN micromamba run uv pip install ./src
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
