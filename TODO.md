# TODOs:
- [ ] Advanced IR synthesis from the histogram
- [ ] Perform benchmarking with pyroomacoustics for thesis
- [ ] Transition to realtime processing
- [ ] Measure latencies and processing delays

## Advanced IR synthesis from the histogram
- [ ] Original Allen & Berkley implementation [1]
- [ ] Sinc interpolation method[2]
- [ ] Frequency domain simulation & phase inversion [3]
- [ ] Energy-decay prediction [4]

## Transition to realtime processing
- [ ] Add portaudio library into the CMake files and build script
- [ ] Attach portaudio callback to the sound source "receive buffer" method
- [ ] Attach portaudio callback to each microphone's "send buffer" method