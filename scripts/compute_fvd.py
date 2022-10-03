import os.path as osp
import glob
import sys
import numpy as np
from tqdm import tqdm
from teco.fvd import fvd

import tensorflow as tf

BATCH_SIZE = 256

path = sys.argv[1]
files = glob.glob(osp.join(path, '*.npz'))
files.sort(key=lambda x: int(osp.basename(x).split('_')[-1].split('.')[0]))
print(f'Found {len(files)} file:', files)

SIZE = np.load(files[0])['real'].shape[0]

def convert(video):
    video = tf.convert_to_tensor(video, dtype=tf.uint8)
    video = tf.cast(video, tf.float32) / 255.
    return video.numpy()

def read(files):
    data = [np.load(f) for f in files]
    data = [(convert(d['real']), convert(d['fake'])) for d in tqdm(data)]
    real, fake = list(zip(*data))
    return real, fake

fvds = []
total = len(files) * SIZE
pbar = tqdm(total=total)
for j in range(0, len(files), BATCH_SIZE // SIZE):
    r, f = read(files[j:j + BATCH_SIZE // SIZE])
    fvds.append(fvd(r, f))
    pbar.update(BATCH_SIZE)
    del r
    del f
print(f'FVD: {np.mean(fvds)} +/- {np.std(fvds)}')
