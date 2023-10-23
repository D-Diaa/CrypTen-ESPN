#!/usr/bin/env bash

python3 scripts/aws_launcher.py \
--ssh_key_file=/home/a2diaa/.ssh/mac,/home/a2diaa/.ssh/mac \
--instances=i-09dfba413e877b802,i-0895a49d766f9ef6c \
--regions=us-west-1,us-east-2 \
scripts/timing_experiments.py