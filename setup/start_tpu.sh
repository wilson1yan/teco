#!/bin/sh

while true
do
	gcloud alpha compute tpus tpu-vm create wilson-v3-8-$1 \
	    --zone=us-central1-a \
	    --accelerator-type='v3-8' \
	    --version='tpu-vm-base'
done