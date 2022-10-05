#! /bin/sh

mkdir -p $1
cd $1

ia download habitat_encoded habitat_encoded.tar
mv habitat_encoded/habitat_encoded.tar .
rmdir habitat_encoded
tar -xf habitat_encoded.tar
rm habitat_encoded.tar