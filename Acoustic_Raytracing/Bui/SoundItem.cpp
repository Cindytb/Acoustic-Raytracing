#pragma once

#include "SoundItem.h"

int SoundItem::num_mics = 0;
int SoundItem::num_src = 0;
int SoundItem::fs = 48000;
OptixShaderBindingTable SoundItem::sbt = {};
OptixTraversableHandle SoundItem::traversable;
OptixPipeline SoundItem::pipeline;


/*
	hist_bin_size : time in seconds - default 0.004
		Amount of time that each histogram bin will accumulate the erngy for
	time_thres : time in seconds - default 10
		Maximum amount of time for the impulse response
	energy_thres : energy/strength of a ray - default 1e-7
		Minimum amount of energy a ray must have before it's terminated
		NOTE: time_thres and energy_thres are NOT equal and are two separate checks for two different room scenarios.
		eg: A very dry room will have rays reaching the energy_thres but not the time_thres.
	dist_thres : distance in meters - default 3430
		dist_thres = c * time_thres
		time threshold converted into a distance.
		With c = 343 and time_thres = 10,
		it will default to 3430
	num_rays : number of rays to shoot out from each sound source
		TBD description for number. Currently testing
	freq_bands : discrete number of frequency bands/bins in the histogram
		freq_bands = (int) log2(fs / 125.0)
		This number is related to the frequency resolution of the histogram.
		Denotes the size of the inner-most dimension of the 2D histogram
		Currently the resolution is set to octave bins starting at 125 Hz until Nyquist.
		With all of the default settings at fs = 48k,
		this defaults to 8
	time_bins : discrete number of time bins in the histogram
		time_bins = fs * time_thres * hist_bin_size
		Denotes the size of the outer dimension of the 2D histogram.
		With all of the default settings at fs = 48k,
		this defaults to 2.5k

	Histogram shape:
			 ________
			|		 |
			|		 |					|
			|		 |					|
			|		 |		Increasing time/distance
			|		 |					|
			|		 |					|
			|		 |					V
			*repeat x2.5k*
			|________|
		--increase fq-->
	The pyroomacoustics implementation that this code is based off inverts the axes,
	so time is the inner-most dimension and frequency is the outer dimension.
	This is swapped in this CUDA/OptiX implementation because it is more cache-friendly.

*/
float SoundItem::hist_bin_size = 0.004;
float SoundItem::time_thres = 5;
float SoundItem::energy_thres = 1e-7;
float SoundItem::dist_thres = 3430;
float SoundItem::c = 343;
int SoundItem::num_rays = 1;
int SoundItem::freq_bands;
int SoundItem::time_bins;


SoundItem::SoundItem(gdt::vec3f pos, gdt::vec3f orientation)//: m_position(pos), m_orientation(orientation)
{
    m_position = pos;
    m_orientation = orientation;
}
SoundItem::SoundItem() : m_position()
{
    gdt::vec3f origin = {0,0,0};
    gdt::vec3f default_orientation = {0,0,0};
    SoundItem(origin, default_orientation);

}