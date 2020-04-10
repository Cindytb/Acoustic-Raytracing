#pragma once

#include "Microphone.h"

Microphone::Microphone()
{
    new (this) Microphone({0, 0, 0}, {0, 0, 0}, 256);
}
Microphone::Microphone(gdt::vec3f pos)
{
    new (this) Microphone(pos, {0, 0, 0}, 256);
}
Microphone::Microphone(gdt::vec3f pos, gdt::vec3f orientation,
                       int frames_per_buffer) : m_frames_per_buffer(frames_per_buffer)
{
    m_position = pos;
    m_orientation = orientation;
    m_output = new float[m_frames_per_buffer];
    m_histogram = new float[time_bins * freq_bands];
    num_mics++;
}
void Microphone::zero_output()
{
    for (int i = 0; i < m_frames_per_buffer; i++)
    {
        m_output[i] = 0.0f;
    }
}

void Microphone::compute_ir(float *d_histogram, cudaStream_t stream)
{
}
Microphone::~Microphone()
{
    num_mics--;
    delete[] m_output;
}