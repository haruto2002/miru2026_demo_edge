#!/bin/bash

pc_name=$1

export DISPLAY=:1
uv run python run.py --cfg pipeline_jetson/config/${pc_name}.yaml