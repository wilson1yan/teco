#! /bin/sh

mkdir -p $1
cd $1

ia download minecraft_marsh_encoded minecraft_encoded.tar
mv minecraft_marsh_encoded/minecraft_encoded.tar .
rmdir minecraft_encoded
tar -xf minecraft_encoded.tar
rm minecraft_encoded.tar