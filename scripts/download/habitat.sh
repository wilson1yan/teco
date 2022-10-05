#!/bin/sh

mkdir -p $1
cd $1

for i in aa ab ac ad
do
    ia download habitat_dataset_$i habitat.tar.part$i
    mv habitat_dataset_$i/habitat.tar.part$i .
    rmdir habitat_dataset_$i
done

cat habitat.tar.part* | tar x

rm habitat.tar.part*
