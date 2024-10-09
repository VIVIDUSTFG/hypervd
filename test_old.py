import json
import time
import numpy as np
import torch


def test_single_video(dataloader, model, args):
    def parse_time(seconds):
        seconds = max(0, seconds)
        sec = seconds % 60
        if sec < 10:
            sec = "0" + str(sec)
        else:
            sec = str(sec)
        return str(seconds // 60) + ":" + sec

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(args.device)
        start_time = time.time()  # Start timer

        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(args.device)
            _, logits, = model(inputs, None)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value[0] > 0.45 else 0 for pred_value in pred]

        video_duration = int(np.ceil(len(pred_binary) * 0.96))

        result = {
            "contains_violence": any(pred == 1 for pred in pred_binary),
            "violence_intervals_seconds": [],
            "violence_intervals_frames": [],
            "video_duration": parse_time(video_duration),
            "analysis_time_elapsed": elapsed_time
        }

        if result["contains_violence"]:
            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    interval_frames = [start_idx, i - 1] if i - \
                        1 != start_idx else [start_idx]
                    interval_seconds = [parse_time(
                        int(np.floor((start_idx + 1) * 0.96))), parse_time(int(np.ceil(i * 0.96)))]
                    result["violence_intervals_frames"].append(interval_frames)
                    result["violence_intervals_seconds"].append(
                        interval_seconds)
                    start_idx = None

            if start_idx is not None:
                interval_frames = [start_idx, len(
                    pred_binary) - 1] if len(pred_binary) - 1 != start_idx else [start_idx]
                interval_seconds = [parse_time(
                    int(np.floor((start_idx + 1) * 0.96))), parse_time(video_duration)]
                result["violence_intervals_frames"].append(interval_frames)
                result["violence_intervals_seconds"].append(interval_seconds)

        return result


def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
