import torch.utils.data as data
import numpy as np
import torch
from preprocess import process_feat


class Dataset(data.Dataset):
    def __init__(self, args, transform=None):
        self.rgb_list_file = args.rgb_list
        self.audio_list_file = args.audio_list

        self.tranform = transform
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.audio_list = list(open(self.audio_list_file))

    def __getitem__(self, index):
        features1 = np.array(
            np.load(self.list[index].strip('\n')), dtype=np.float32)
        # features2 = np.array(
        #    np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
        features2 = np.array(
            np.load(self.audio_list[index].strip('\n')), dtype=np.float32)
        if features1.shape[0] == features2.shape[0]:
            features = np.concatenate((features1, features2), axis=1)
        else:
            min_length = min(features1.shape[0], features2.shape[0])
            features1 = features1[:min_length, :]
            features2 = features2[:min_length, :]
            features = np.concatenate((features1, features2), axis=1)

        if self.tranform is not None:
            features = self.tranform(features)

        return features

    def __len__(self):
        return len(self.list)
