import argparse
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size

def get_weight_size(path):
    stats = Path(path).stat()
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/LAO/weights/best.pt', help='Path to weights file')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--warmup', type=int, default=200, help='Warmup iterations')
    parser.add_argument('--testtime', type=int, default=1000, help='Test iterations')
    parser.add_argument('--half', action='store_true', default=False, help='Use FP16 precision')

    opt = parser.parse_args()

    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'

    # Load model
    model = attempt_load(opt.weights, map_location=device)
    model.eval()
    if half:
        model.half()

    # Input image
    img = torch.randn(opt.batch, 3, opt.imgsz, opt.imgsz).to(device)
    if half:
        img = img.half()

    # Warmup
    print('Warming up...')
    for _ in tqdm(range(opt.warmup), desc='Warmup'):
        _ = model(img, augment=False)[0]

    # Test FPS
    print('Testing latency...')
    time_arr = []
    for _ in tqdm(range(opt.testtime), desc='Testing'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(img, augment=False)[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)

    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    fps = 1.0 / infer_time_per_image

    print(f'Model weights: {opt.weights}, size: {get_weight_size(opt.weights)} MB')
    print(f'Batch size: {opt.batch}, Image size: {opt.imgsz}x{opt.imgsz}')
    print(f'Latency per image: {infer_time_per_image:.6f}s ± {std_time:.6f}s, FPS: {fps:.2f}')
