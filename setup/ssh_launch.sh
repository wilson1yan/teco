#! /bin/bash

SCRIPT_DIR="setup"

COMMAND=$1

shift

for host in "$@"; do
    echo "Launching on host: $host"
    ssh $host 'tmux new -d '"$COMMAND" &
done

wait
