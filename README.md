<center>

# OptiX 7 Powered Realtime Acoustic Raytracing

## Cindy Bui
Submitted in partial fulfillment of the requirements for the 

Master of Music in Music Technology 

Department of Music and Performing Arts Professions

Steinhardt School of Culture, Education, and Human Development

New York University
</center>

# About this Repository
This repository initially came from [Ingo Wald's 2019 Siggraph course materials](https://github.com/ingowald/optix7course). 
All of the setup and CMake scripts came from him. 

This repo utilizes OptiX 7 for acoustic raytracing and serves as the 
project for my thesis. The raytracing implementation was inspired
by [pyroomacoustics](https://github.com/LCAV/pyroomacoustics). At the 
time of writing and programming, there is a beta branch called 
"next-gen-simulator" which contains a Cython raytracing 
implementation. This implementation was benchmarked against that 
implementation and passes with up to 5e-7 precision. Specifically, the
raytracing histograms were benchmarked. [This](https://github.com/Cindytb/pyroomacoustics) is my fork of pyroomacoustics
for the benchmarking measurements.

Building it is almost exactly the same as the optix7course. I added
the cuda helper files from the CUDA Samples into common/cuda_helpers. 

# Building the Code

This code was intentionally written with minimal dependencies,
requiring only CMake (as a build system), your favorite
compiler (tested with Visual Studio 2017 and 2019 under Windows, and GCC under
Linux), and the OptiX 7 SDK (including CUDA 10.1 and NVIDIA driver recent
enough to support OptiX).

## Dependencies

- a compiler
    - On Windows, tested with Visual Studio 2017 and 2019 community editions
    - On Linux, tested with Ubuntu 18 and Ubuntu 19 default gcc installs
- CUDA 10.1
    - Download from developer.nvidia.com
    - on Linux, suggest to put `/usr/local/cuda/bin` into your `PATH`
- latest NVIDIA developer driver that comes with the SDK
    - download from http://developer.nvidia.com/optix and click "Get OptiX"
- OptiX 7 SDK
    - download from http://developer.nvidia.com/optix and click "Get OptiX"
    - on linux, suggest to set the environment variable `OptiX_INSTALL_DIR` to wherever you installed the SDK.  
    `export OptiX_INSTALL_DIR=<wherever you installed OptiX 7.0 SDK>`
    - on windows, the installer should automatically put it into the right directory

Detailed steps below:

## Building under Linux

- Install required packages

    - on Debian/Ubuntu: `sudo apt install cmake-curses-gui`
    - on RedHat/CentOS/Fedora (tested CentOS 7.7): `sudo yum install cmake3`

- Clone the code
```
    git clone git@github.com:Cindytb/Acoustic_Raytracing.git
    cd Acoustic_Raytracing
```

- create (and enter) a build directory
```
    mkdir build
    cd build
```

- configure with cmake
    - Ubuntu: `cmake ..`
    - CentOS 7: `cmake3 ..`

- and build
```
    make
```

## Building under Windows

- Install Required Packages
	- see above: CUDA 10.1, OptiX 7 SDK, latest driver, and cmake
- download or clone the source repository
- Open `CMake GUI` from your start menu
	- point "source directory" to the downloaded source directory
	- point "build directory" to <source directory>/build (agree to create this directory when prompted)
	- click 'configure', then specify the generator as Visual Studio 2017 or 2019, and the Optional platform as x64. If CUDA, SDK, and compiler are all properly installed this should enable the 'generate' button. If not, make sure all dependencies are properly installed, "clear cache", and re-configure.
	- click 'generate' (this creates a Visual Studio project and solutions)
	- click 'open project' (this should open the project in Visual Studio)

## Bibliography
### Impulse Response Synthesis Methods
[1]	J. Allen and D. Berkley, Image method for efficiently simulating small-room acoustics, Journal of the Acoustical Society of America, vol. 65(4), pp. 943-950, April 1979.

[2]	P. Peterson, Simulating the response of multiple microphones to a single acoustic source in a reverberant room, Journal of the Acoustical Society of America, vol. 80(5), pp. 1527–1529, November 1986.

[3]	E. Lehmann and A. Johansson, Prediction of energy decay in room impulse responses simulated with an image-source model, Journal of the Acoustical Society of America, vol. 124(1), pp. 269-277, July 2008.

[4]	E. Lehmann, A. Johansson, and S. Nordholm, Reverberation-time prediction method for room impulse responses simulated with the image-source model, Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA'07), pp. 159-162, New Paltz, NY, USA, October 2007.

### GPGPU
Owens, J. D., Luebke, D., Govindaraju, N., Harris, M., Krüger, J., Lefohn, A. E., & Purcell, T. J. (2007). A Survey of General-Purpose Computation on Graphics Hardware. Computer Graphics Forum, 26(1), 80–113. https://doi.org/10.1111/j.1467-8659.2007.01012.x

Hoshino, T., Maruyama, N., Matsuoka, S., & Takaki, R.  (2013, May).   CUDA vs. OpenACC:  Performance Case Studies with Kernel Benchmarks and a Memory-Bound CFD Application. In the 2013 13th IEEE/ACM International Symposium on Cluster, Cloud, and Grid Computing (p.  136-143).   doi:  10.1109/CCGrid.2013.12

Faella, J. (2013). On Performance of GPU and DSP Architectures for Computationally Intensive Applications. (Master’s thesis, University of Rhode Island). Retrieved from Open Access Master’s Theses. Paper 2. http://digitalcommons.uri.edu/theses/2

Savioja, L., Valimaki, V., & Smith, J. O. (2011). Audio signal processing using graphics processing units. J. Audio Eng. Soc, 59 (1/2), 3-19. 

Tsingos, N. (2009, Oct). Using Programmable Graphics Hardware for Acoustics and Audio Rendering. Proceedings of the 127th Audio Engineering Society Convention.

Hsu, B., & Sosnick-Perez, M. (2013, April). Realtime GPU audio. ACMQueue, 11 (4), 40:40-40:55. 

Belloch, J. A., Gonzalez, A., Martínez-Zaldívar, F.J., & Vidal, A.M, (2011, Dec 01). Real-time massive convolution for audio applications on GPU. The Journal of Supercomputing, 48(3), 449-457.

Belloch, J. A., Gonzalez, A., Martinez-Zaldivar, F. J., & Vidal, A. M. (2011, July). A Real-Time Crosstalk Canceller on a notebook GPU. In 2011 IEEE International Conference on Multimedia and Expo (p. 1-4). doi: 10.1109/ICME.2011.6012072

Belloch, J. A., Ferrer, M., Gonzalez, A., Martinez-Zaldivar, F. J., & Vidal, A. M. (2013). Headphone-based virtual spatialization of sound with a GPU accelerator. Journal of the Audio Engineering Society, 61 (7/8), 546-561.

Belloch, J. A., Bank, B., Savioja, L., Gonzalez, A., & Välimäki, V. (2014, May). Multi-channel IIR filtering of audio signals using a GPU. Proceedings of the 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 6692-6696. doi: 10.1109/ICASSP.2014.6854895


### Artificial Reverberation
Schroeder, M. R. (1962). Natural sounding artificial reverberation. AES Journal of the Audio Engineering Society: Audio, Acoustics, Applications, 10, 219–223.


Griesinger, D. (1989, May). Practical processors and programs for digital reverberation. In Audio engineering society conference: 7th international conference: Audio in digital times. 


#### Geometry Based
Allen J. B, Berkley D. A (1978, May) Image method for efficiently simulating small-room acoustics. The Journal of the Acoustical Society of America, 65(4), 943-950. Doi:10.1121/1.382599

Lehmann,  E.  A.,  &  Johansson,  A.  M.   (2008).   Prediction  of  energy  decay  in  room  impulse  responsessimulated  with  an  image-source  model.The  Journal  of  the  Acoustical  Society  of  America,124(1),269-277.  Retrieved fromhttps://doi.org/10.1121/1.2936367doi:  10.1121/1.2936367

Peterson, P. M.  (1986).  Simulating the response of multiple microphones to a single acoustic source in areverberant room.The Journal of the Acoustical Society of America,80(5), 1527-1529. Retrieved fromhttps://doi.org/10.1121/1.394357doi:  10.1121/1.394357

Savioja, L., Svensson U. P. (2015, August). Overview of Geometrical Room Acoustic Modeling Techniques. Journal of the Acoustic Society of America. 138(2), 708-730. Retrieved from https://doi.org/10.1121/1.4926438 doi:  10.1121/1.4926438


#### Statistical
Stephenson, U. M. (1989, March). The Sound Particle Simulation Technique-An Efficient Prediction Method for Room Acoustical Parameters of Concert Halls. Audio Engineering Society Convention 86. Retrieved from http://www.aes.org/e-lib/browse.cfm?elib=5942


Spring N.F., Gilford C.L.S. (1974, March) Artificial Reverberation. BBC Engineering 97 Mar. 1974 pp 28-32.


### Practical Implementations
Schissler C. & Manocha, D. (2011, Feb.) Gsound: Interactive sound propagation for games. In Audio Engineering Society Conference: 41st interational conference: Audio for games.

Jot, J.-M., & Lee, K. S. (2016, Sep). Augmented reality headphone environment rendering. In Audio Engineering Society Conference: 2016 AES international conference on audio for virtual and augmented reality. Retrieved from http://www.aes.org/e-lib/browse.cfm?elib=18506

### GPGPUs and Reverb
Jedrzejewski, M., & Marasek, K. (2006). Computation of Room Acoustics Using Programmable Video Hardware. In K. Wojciechowski, B. Smolka, H. Palus, R. S. Kozera, W. Skarbek, & L. Noakes (Eds.), Computer Vision and Graphics: International Conference, ICCVG 2004, Warsaw, Poland, September 2004, Proceedings (pp. 587-592). Dordrecht: Springer Netherlands. doi:10.1007/1-4020-4179-9_84


Röber, N., Kaminski, U., & Masuch, M. (0002, 10). Ray acoustics using computer graphics technology. Proceedings of the 10th International Conference on Digital Audio Effects, DAFx 2007.


Fu, Z.-h., & Li, J.-w. (2016, 5 01). GPU-based image method for room impulse response calculation. Multimedia Tools and Applications, 75, 5205-5221. doi:10.1007/s11042-015-2943-4


Nikolov V. D., Misic, M. J., & Tomasevic, M.V (2015, Nov). GPU-based Implementation of Reverb Effect. In 2015 23rd telecommunications form telfor (telfor) (p. 990-993). Doi: 10.1109/TELFOR.2015.7377634

