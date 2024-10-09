from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import test_single_video, save_results
import option
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test():
    print('perform testing...')
    args = option.parser.parse_args()
    args.device = 'cuda:' + \
        str(args.cuda) if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(Dataset(args), batch_size=1,
                             shuffle=False, num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(args.device)

    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(Path(__file__).resolve().parent, 'ckpt/pretrained.pkl'), map_location=torch.device('cpu')).items()})
    # st = time.time()

    results = test_single_video(
        test_loader, model, args)
    save_results(results, os.path.join(
        args.output_path, 'results.npy'))


if __name__ == '__main__':
    test()
