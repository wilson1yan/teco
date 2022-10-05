#!/bin/sh

mkdir -p $1
cd $1

for i in aa ab ac ad ae af ag ah ai aj ak
do
    ia download minecraft_marsh_dataset_$i minecraft.tar.part$i
    mv minecraft_marsh_dataset_$i/minecraft.tar.part$i .
    rmdir minecraft_marsh_dataset_$i
done

cat minecraft.tar.part* | tar x

rm minecraft.tar.part*
