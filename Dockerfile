FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
# Deal with numba
RUN mkdir -p /tmp/numba_cache & chmod 777 /tmp/numba_cache & NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache


# Install package
# Copy your source code into the image
ADD . /app
WORKDIR /app
RUN uv pip install --system .

# Check install was successful
RUN beamcon_3D -h
CMD ["bash"]
