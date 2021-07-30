#!/bin/sh

set -e

tensorboard --logdir logs --host 0.0.0.0 --port 6006 &
python3 src/main.py && fg