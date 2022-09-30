#! /bin/bash

SCRIPT_DIR="setup"

for host in "$@"; do
	rsync -r . $host:~/teco &
done

wait