# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS, Auto) # optional argument for arch to add
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61)
include_directories(${OptiX_INCLUDE} ${PROJECT_SOURCE_DIR}/common/cuda_helpers ${SNDFILE_INCLUDE_DIR} ${PORTAUDIO_INCLUDE_DIR} /usr/local/include)
link_directories(${SNDFILE_INCLUDE_DIR} ${PORTAUDIO_INCLUDE_DIR} /usr/local/lib)
cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

cuda_add_executable(Acoustic_Raytracing.out
  audio.cpp
  audio.h
  corelink.cpp
  corelink.h
  kernels.cu
  kernels.cuh
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  debug.cuh
  kernels.cuh
  LaunchParams.h
  main.cpp
  prd.h
  OptixSetup.h
  OptixSetup.cpp
  Bui/Microphone.h
  Bui/Microphone.cpp
  Bui/SoundItem.h
  Bui/SoundItem.cpp
  Bui/SoundSource.h
  Bui/SoundSource.cpp
  Bui/convolve.cu
  Bui/convolve.cuh
  )
target_link_libraries(Acoustic_Raytracing.out
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  sndfile.so
  ${CUDA_cufft_LIBRARY}
  Corelink.so
  CorelinkWrapper.so
  )
