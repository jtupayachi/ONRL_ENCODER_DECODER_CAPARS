FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages
RUN pip3 install --no-cache-dir \
    pandas \
    numpy \
    openpyxl \
    pyarrow \
    torch \
    scikit-learn \
    matplotlib

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["tail", "-f", "/dev/null"]
