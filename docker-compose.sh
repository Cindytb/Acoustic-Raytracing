#!/bin/bash

# Build the current repo into a docker image
docker build -t cindy/acoustic-raytracing:0.0.1 .

# Run the docker 
docker run --gpus all -it -v $(pwd):/code cindy/acoustic-raytracing:0.0.1 bash -c "mkdir -p /code/build & cd /code/build && cmake .. && make && ./Acoustic_Raytracing.out"