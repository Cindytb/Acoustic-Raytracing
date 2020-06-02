FROM corelink/cuda-10.2-devel-ubuntu18.04:0.0.1

# Export the OptiX installation directory
COPY NVIDIA-OptiX-SDK-7.0.0-linux64 /opt/optix
ENV OptiX_INSTALL_DIR /opt/optix
ENV LD_LIBRARY_PATH /opt/optix/lib64:${LD_LIBRARY_PATH}
# ENV NVIDIA_DRIVER_CAPABILITIES graphics
# Installing project-specific libraries
RUN apt install -y libsndfile1-dev