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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(Acoustic_Raytracing)

cmake_minimum_required(VERSION 2.8)
if (NOT WIN32)
# visual studio doesn't like these (not need them):
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
set (SNDFILE_LIB sndfile.so)
set (PORTAUDIO_LIB libportaudio.so)
set (SNDFILE_INCLUDE_DIR /usr/include) # For standalone linux
set (PORTAUDIO_INCLUDE_DIR /usr/include) # For standalone linux
else(WIN32)
set (SNDFILE_LIB libsndfile-1.lib)
set (PORTAUDIO_LIB portaudio_x64.lib)
set (SNDFILE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/common/libsndfile) # For windows, put precompiled .lib in repo
set (PORTAUDIO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/common/portaudio) # For windows, put precompiled .lib in repo
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

add_subdirectory(Acoustic_Raytracing)