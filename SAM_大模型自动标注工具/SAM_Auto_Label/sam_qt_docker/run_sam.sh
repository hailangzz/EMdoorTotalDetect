#!/bin/bash
xhost +local:docker

docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME:/workspace \
  sam-qt \
  conda run -n samqt python segment_anything_annotator.py "$@"

