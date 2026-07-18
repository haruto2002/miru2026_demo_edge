#!/bin/bash

pc_name=$1
display=$2

export DISPLAY=:1
args=(--cfg "pipeline_jetson/config/${pc_name}.yaml")
if [[ -n "${display}" ]]; then
  args+=(--display "${display}")
fi
uv run python run.py "${args[@]}"
