#!/usr/bin/env bash

# Use with caution
yes | rm -r output/runs/*
yes | rm -r output/checkpoint-*
yes | rm ./fine-tune-gpu.cid
docker rm $(docker ps -aq --filter "ancestor=fine-tune-gpu")
