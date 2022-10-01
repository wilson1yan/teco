#!/bin/sh

while true
do
	gcloud alpha compute tpus tpu-vm create wilson-v3-$2-$1 \
	    --zone=us-east1-d \
	    --accelerator-type="v3-$2" \
	    --version='tpu-vm-base' --preemptible
done
