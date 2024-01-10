#!/usr/bin/env bash

python3 scripts/aws_launcher.py \
--ssh_key_file=/home/.ssh/mac,/home/.ssh/mac \
--instances=i-05c9490a17d3acb9e,i-0f5e5547b00fdced1 \
--regions=eu-central-1,us-east-2 \
scripts/timing_experiments.py