#! /bin/sh

mkdir -p $1
cd $1

ia download dmlab_encoded dmlab_encoded.tar
mv dmlab_encoded/dmlab_encoded.tar .
rmdir dmlab_encoded
tar -xf dmlab_encoded.tar
rm dmlab_encoded.tar