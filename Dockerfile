# VecDB Development Container
# Ubuntu-based image with C++17, Python 3.11, CMake, pybind11

FROM ubuntu:22.04

LABEL maintainer="VecDB Project"
LABEL description="Development environment for VecDB vector database"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # C++ toolchain
    build-essential \
    g++ \
    cmake \
    ninja-build \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Development tools
    git \
    curl \
    wget \
    vim \
    # Debugging tools
    gdb \
    valgrind \
    # For pybind11
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install pybind11 (both pip and system-wide for CMake discovery)
RUN pip install pybind11[global] \
    && pip install pybind11-stubgen

# Install Python dependencies (runtime)
RUN pip install --no-cache-dir \
    numpy>=1.20

# Install Python dependencies (development)
RUN pip install --no-cache-dir \
    pytest>=7.0 \
    pytest-cov>=4.0 \
    pytest-xdist \
    black \
    isort \
    mypy

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaces/vecdb

# Switch to non-root user
USER $USERNAME

# Add local bin to PATH for user-installed packages
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Default command
CMD ["/bin/bash"]
