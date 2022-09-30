#! /bin/bash

cat > $HOME/.ssh/config << EOF
Host github.com
  StrictHostKeyChecking no
EOF

git clone git@github.com:wilson1yan/teco
cd hier_video

pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r tpu_requirements.txt
pip3 install -e .

touch $HOME/init.sh
echo 'export WANDB_API_KEY=b9b582bde5d3823c9e83ecf0db78e627f574628d' >> $HOME/init.sh
echo 'export DATA_DIR=/home/wilson/logs/hier_video' >> $HOME/init.sh
echo 'export WANDB_API_KEY=b9b582bde5d3823c9e83ecf0db78e627f574628d' >> $HOME/.bashrc
echo 'export DATA_DIR=/home/wilson/logs/hier_video' >> $HOME/.bashrc

git config --global user.email "wilson1.yan@berkeley.edu"
git config --global user.name "Wilson Yan"

sudo apt-get update && sudo apt-get install -y \
	ffmpeg