import json
import time
import numpy as np
import torch


def test_single_video(dataloader, model, args):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)

        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

        pred = pred.cpu().detach().numpy()
        pred_binary = np.array([1 if pred_value[0] > 0.45 else 0 for pred_value in pred])
        return pred_binary


def save_results(results, filename):
    np.save(filename, results)
