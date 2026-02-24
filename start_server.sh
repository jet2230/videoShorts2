#!/bin/bash
# Script to start the Video Shorts Creator server from any directory
PROJECT_DIR="/home/obo/playground/videoShorts2"
cd "$PROJECT_DIR" || exit
python3 server.py
