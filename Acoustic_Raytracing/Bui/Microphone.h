#pragma once

#include "SoundItem.h"
#include "gdt/math/vec.h"

/*CUDA Includes*/

#include <cuda_runtime.h>
class Microphone : SoundItem
{
public:
    Microphone();
    Microphone(gdt::vec3f pos);
    Microphone(gdt::vec3f pos, gdt::vec3f orientation, int frames_per_buffer);
    void compute_ir(float *d_histogram, cudaStream_t stream);
    void zero_output();
    ~Microphone();

private:
    float *m_output;
    float *m_ir;
    float *m_histogram;
    int m_frames_per_buffer;
};