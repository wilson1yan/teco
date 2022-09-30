#! /bin/bash

SCRIPT_DIR="setup"


for host in "$@"; do
    scp -o "StrictHostKeyChecking no" $SCRIPT_DIR/tpu_vm_setup.sh wilson@$host:~/
    ssh -o "StrictHostKeyChecking no" $host 'sh tpu_vm_setup.sh' &
done

wait