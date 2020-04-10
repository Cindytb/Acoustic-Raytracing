#include "audio.h"

PaStream* stream;

void initializePA(int fs, osc::SampleRenderer* renderer) {
	PaError err;
	/*PortAudio setup*/
	PaStreamParameters outputParams;
	PaStreamParameters inputParams;

	/* Initializing PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
		printf("PortAudio error: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Input stream parameters */
	inputParams.device = Pa_GetDefaultInputDevice();
	inputParams.channelCount = 1;
	inputParams.sampleFormat = paFloat32;
	inputParams.suggestedLatency =
		Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
	inputParams.hostApiSpecificStreamInfo = NULL;

	/* Ouput stream parameters */
	outputParams.device = Pa_GetDefaultOutputDevice();
	outputParams.channelCount = 2;
	outputParams.sampleFormat = paFloat32;
	outputParams.suggestedLatency =
		Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
	outputParams.hostApiSpecificStreamInfo = NULL;

	/* Open audio stream */
	err = Pa_OpenStream(&stream,
		&inputParams,
		&outputParams,
		fs, FRAMES_PER_BUFFER,
		paNoFlag, /* flags */
		paCallback,
		renderer);
	if (err != paNoError) {
		printf("PortAudio error: open stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: open stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Start audio stream */
	err = Pa_StartStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: start stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: start stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

}

void closePA() {
	PaError err;
	/* Stop stream */
	err = Pa_StopStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Close stream */
	err = Pa_CloseStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Terminate PortAudio */
	err = Pa_Terminate();
	if (err != paNoError) {
		printf("PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}
}
static int paCallback(const void* inputBuffer, void* outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void* userData)
{
	/* Cast data passed through stream to our structure. */
	osc::SampleRenderer* renderer = (osc::SampleRenderer*)userData;
	float* output = (float*)outputBuffer;
	float* input = (float*)inputBuffer;

	// Sending input directly back to output
	// stereo output is interleaved
	for (unsigned i = 0; i < framesPerBuffer; i++) {
		output[i * 2] = input[i];
		output[i * 2 + 1] = input[i];
	}
	return 0;
}

