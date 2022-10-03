import sys
import numpy as np
from tqdm import tqdm
import glob
import os.path as osp

from teco.metrics import get_ssim, get_psnr, get_lpips

BATCH_SIZE = 256

path = sys.argv[1]
if path.endswith('/'):
    path = path[:-1]
open_loop_ctx = int(osp.basename(path).split('_')[-1])

files = glob.glob(osp.join(path, '*.npz'))
files.sort(key=lambda x: int(osp.basename(x).split('_')[-1].split('.')[0]))
print(f'Found {len(files)} file:', files)

SIZE = np.load(files[0])['real'].shape[0]

def read(files):
    scale = np.array(255., dtype=np.float32)
    data = [np.load(f) for f in files]
    data = [(d['real'][:, open_loop_ctx:] / scale, d['fake'][:, open_loop_ctx:] / scale) for d in data]
    return data

ssim_fn = get_ssim()
psnr_fn = get_psnr()
lpips_fn = get_lpips()

ssims, psnrs, lpips = [], [], []
total = len(files) * SIZE
pbar = tqdm(total=total)
for j in range(0, len(files), BATCH_SIZE // SIZE):
    data = read(files[j:j + BATCH_SIZE // SIZE])
    ps, ss, ls = [], [], []
    for r_i, f_i in data:
        ps.append(psnr_fn(r_i, f_i).mean())
        ss.append(ssim_fn(r_i, f_i).mean())
        ls.append(lpips_fn(r_i, f_i).mean())
        pbar.update(r_i.shape[0])
    psnrs.append(np.mean(ps))
    ssims.append(np.mean(ss))
    lpips.append(np.mean(ls))

print(f'PSNR: {np.mean(psnrs)} +/- {np.std(psnrs)}')
print(f'SSIM: {np.mean(ssims)} +/- {np.std(ssims)}')
print(f'LPIPS: {np.mean(lpips)} +/- {np.std(lpips)}')
