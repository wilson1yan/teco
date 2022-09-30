#!/bin/sh

TPU_IPS=$(gcloud alpha compute tpus tpu-vm describe $1 --zone=$2 --format="value[delimiter=' '](networkEndpoints.accessConfig.externalIp)")
echo $TPU_IPS