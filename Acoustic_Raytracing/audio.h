#pragma once
#ifndef _AUDIO_H_
#define _AUDIO_H_

#include "SampleRenderer.h"
#include "Bui/constants.h"

#include <portaudio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int paCallback(const void* inputBuffer, void* outputBuffer,
  unsigned long framesPerBuffer,
  const PaStreamCallbackTimeInfo* timeInfo,
  PaStreamCallbackFlags statusFlags,
  void* userData);
void initializePA(int fs, osc::SampleRenderer* renderer);
void closePA();


#endif
