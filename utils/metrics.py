__author__ = 'yunbo'

import numpy as np

def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1,2), dtype=np.float32)
    return np.mean(mae)

def batch_psnr(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y)**2, axis=(1,2), dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

