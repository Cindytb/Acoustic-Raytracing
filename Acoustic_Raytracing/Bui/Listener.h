#pragma once

#include "Microphone.h"
#include <iostream>
#include <vector>

class Listener : SoundItem
{
    /* TODO: For future use */
    float *hrtf;
    float *fft_hrtf;

    Microphone left;
    Microphone right;
};