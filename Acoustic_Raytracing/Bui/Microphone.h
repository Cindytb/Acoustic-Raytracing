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
    void zero_output();
    void attach_output(float* output);
    float* get_output();
    ~Microphone();

private:
    float *m_output;
};