FROM corelink/cuda-10.2-devel-ubuntu18.04:0.0.1

# Export the OptiX installation directory
ADD NVIDIA-OptiX-SDK-7.0.0-linux64 /opt/optix
ADD OptiX_INSTALL_DIR /opt/optix

# Installing project-specific libraries
RUN apt install -y libsndfile1-dev

# Testing the build. This does not actually build it
RUN mkdir -p /code/build
WORKDIR /code/build
RUN cmake ..
RUN make