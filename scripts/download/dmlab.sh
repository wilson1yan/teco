#!/bin/sh

mkdir -p $1
cd $1

for i in aa ab ac
do
    ia download dmlab_dataset_$i dmlab.tar.part$i
    mv dmlab_dataset_$i/dmlab.tar.part$i .
    rmdir dmlab_dataset_$i
done

cat dmlab.tar.part* | tar x

rm dmlab.tar.part*
