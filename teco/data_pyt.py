import os.path as osp
import warnings
import glob
import pickle
import math
import h5py
import numpy as np
import json

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.io import read_video


class Data:
    def __init__(self, config):
        self.config = config
    
    def _dataset(self, train):
        if self.config.data_path.endswith('.hdf5'):
            Dataset = HDF5Dataset
        elif 'gqn_mazes' in self.config.data_path or 'dl_maze' in self.config.data_path:
            Dataset = NumpyDataset
        else:
            Dataset = VideoDataset
        dataset = Dataset(self.config, train=train)
        return dataset
    
    def _dataloader(self, train):

        dataset = self._dataset(train)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def test_dataloader(self):
        return self._dataloader(False)
 

class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-1, 1] """
    def __init__(self, config, train=True):
        """
        Args:
            args.data_path: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64,
                }
            args.sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.config = config
        self.seq_len = config.seq_len if train else config.eval_seq_len

        # read in data
        self.data_file = config.data_path
        self.data = h5py.File(self.data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        if f'{self.prefix}_actions' in self.data:
            self._actions = self.data[f'{self.prefix}_actions'][:]
        self._idx = self.data[f'{self.prefix}_idx'][:]
        self.size = len(self._idx)

    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        if end - start > self.seq_len:
            start = start + np.random.randint(low=0, high=end - start - self.seq_len)
        assert start < start + self.seq_len <= end, f'{start}, {end}'
        video = torch.tensor(self._images[start:start + self.seq_len]) 
        video = preprocess(video, self.config.image_size)

        out = dict(video=video)
        if f'{self.prefix}_actions' in self.data: 
            actions = torch.tensor(self._actions[start:start + self.seq_len - 1])
            actions = torch.LongTensor(actions) + 1
            actions = F.pad(actions, (1, 0))
            out['actions'] = actions
        return out


class VideoDataset(data.Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config
        self.seq_len = config.seq_len

        if 'kinetics600_subset' in config.data_path:
            self.video_paths = json.load(open(osp.join(config.data_path, 'paths2.json'), 'r'))
            self.video_paths = [osp.join(config.data_path, p) for p in self.video_paths]
        else:
            folder = osp.join(config.data_path, 'train' if train else 'test')
            files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                         for ext in self.exts], [])
            self.video_paths = files
            self.video_paths.sort()

            if 'kinetics' in config.data_path:
                def get_file_end(name):
                    return osp.join(osp.basename(osp.dirname(name)), osp.basename(name))

                ignore_files = json.load(open('/home/wilson/data/kinetics600/bad_paths.json', 'r'))
                ignore_files = set(ignore_files)
                original_len = len(self.video_paths)
                self.video_paths = list(filter(lambda x: get_file_end(x) not in ignore_files, self.video_paths))
                print(f'Filtered ignored files {len(self.video_paths)} / {original_len}')
            
            rng = np.random.default_rng(0)
            rng.shuffle(self.video_paths)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video = read_video(video_path, pts_unit='sec')[0]
        video = video[:self.seq_len]
        video = preprocess(video, self.config.image_size)
        out = dict(video=video)

        #action_path = video_path[:-3] + 'npz'
        action_path = video_path[:-3] + 'npy'
        if osp.exists(action_path):
            #actions = np.load(action_path)['actions']
            actions = np.load(action_path)
            out['actions'] = actions[:self.seq_len]

        return out

class NumpyDataset(data.Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """

    def __init__(self, config, train=True):
        super().__init__()
        self.train = train
        self.config = config
        self.seq_len = config.seq_len

        folder = osp.join(config.data_path, 'train' if train else 'test')
        files = glob.glob(osp.join(folder, '**', f'*.npz'), recursive=True)
        self.video_paths = files
        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        data = np.load(video_path)

        video = data['video']
        video = torch.from_numpy(video)
        video = video[:self.seq_len]
        video = preprocess(video, self.config.image_size)
        out = dict(video=video)

        if 'actions' in data:
            out['actions'] = torch.LongTensor(data['actions'])

        return out
        
def preprocess(video, image_size, sequence_length=None):
    # video: THWC, {0, ..., 255}

    resolution = image_size

    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    video = 2 * video - 1 # [0, 1] -> [-1, 1]
    video = video.movedim(1, -1)
    return video
