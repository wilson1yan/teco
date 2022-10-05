#! /bin/sh

mkdir -p $1
cd $1

ia download kinetics600_encoded kinetics600_encoded.tar
mv kinetics600_encoded/kinetics600_encoded.tar .
rmdir kinetics600_encoded
tar -xf kinetics600_encoded.tar
rm kinetics600_encoded.tar