#pragma once

#include "Microphone.h"

Microphone::Microphone()
{
    new (this) Microphone({0, 0, 0}, {0, 0, 0}, FRAMES_PER_BUFFER);
}
Microphone::Microphone(gdt::vec3f pos)
{
    new (this) Microphone(pos, {0, 0, 0}, FRAMES_PER_BUFFER);
}
Microphone::Microphone(gdt::vec3f pos, gdt::vec3f orientation,
                       int frames_per_buffer)
{
    m_position = pos;
    m_orientation = orientation;
    SoundItem::frames_per_buffer = frames_per_buffer;
    num_mics++;
}
void Microphone::zero_output()
{
    for (int i = 0; i < frames_per_buffer; i++)
    {
        m_output[i] = 0.0f;
    }
}

float* Microphone::get_output(){
    return m_output;
}

void Microphone::attach_output(float* output){
    m_output = output;
    zero_output();
}
Microphone::~Microphone()
{
    num_mics--;
    delete[] m_output;
}