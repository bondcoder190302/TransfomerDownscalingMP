import torch
import numpy as np
import os
import math
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from .climatesr_model import ClimateSRAddHGTModel
from basicsr.utils.registry import MODEL_REGISTRY
from torch import distributed as dist


# =====================================================
# Utility: pad image to nearest multiple of window size
# =====================================================
def expand2square2(timg, factor=16.0):
    """Pad tensor (4D or 5D) to be divisible by given factor using reflection padding."""
    h, w = timg.shape[-2:]
    rsp = timg.shape[:-2]
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    mod_pad_w = X - w
    mod_pad_h = X - h
    if timg.ndim == 5:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h, 0, 0), mode='reflect')
    elif timg.ndim == 4:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
    else:
        raise ValueError(f"Unexpected tensor ndim: {timg.ndim}")
    mask = torch.zeros(rsp[0], 1, X, X, device=timg.device, dtype=timg.dtype)
    mask[..., :h, :w] = 1.0
    return img, mask


# =====================================================
# Uformer Model - Single Variable Precipitation version
# =====================================================
@MODEL_REGISTRY.register()
class ClimateUformerModel(ClimateSRAddHGTModel):
    """Simplified single-scale Climate Uformer (used for baseline precipitation version)."""

    def test(self):
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 2)

        # prepare low-res input
        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None

        # prepare high-res auxiliary (elevation)
        if 'hgt' in self.lq and self.lq['hgt'].numel() > 0:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            self.output = net(self.lq, mask if hasattr(self, 'net_g_ema') else None)
        self.output = self.output[..., :h * scale, :w * scale]
        net.train()


# =====================================================
# Multi-Scale Fusion version (Main training model)
# =====================================================
@MODEL_REGISTRY.register()
class ClimateUformerMultiscaleFuseModel(ClimateSRAddHGTModel):
    """
    Modified version for precipitation downscaling with elevation as auxiliary input.
    Only one input (precip) and one output channel.
    """

    def test(self):
        """Inference: pad to multiples of window size, apply model, crop back."""
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 2)

        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None

        # elevation (optional)
        if 'hgt' in self.lq and self.lq['hgt'].numel() > 0:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            self.output = net(self.lq, mask if hasattr(self, 'net_g_ema') else None)

        # first element in output list is main HR prediction
        if isinstance(self.output, (list, tuple)):
            self.output = self.output[0]

        self.output = self.output[..., :h * scale, :w * scale]
        net.train()

    # =====================================================
    # Training Step
    # =====================================================
    def optimize_parameters(self, current_iter):
        """Forward + loss + backward for precipitation regression."""
        self.optimizer_g.zero_grad(set_to_none=True)

        # Forward pass (with AMP support)
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.output = self.net_g(self.lq)
        if isinstance(self.output, (list, tuple)):
            self.output = [v.float() for v in self.output]
        else:
            self.output = [self.output.float()]

        l_total = 0.0
        loss_dict = OrderedDict()
        scale = self.opt.get('scale', 2)

        # --------------------
        # 1) Main pixel loss
        # --------------------
        if self.cri_pix:
            l_pix = self.cri_pix(self.output[0], self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # --------------------
        # 2) Optional multi-scale auxiliary loss
        # --------------------
        if self.cri_pix_multiscale and len(self.output) > 1:
            gt_levels = [
                F.interpolate(self.gt, scale_factor=1 / (scale * 4), mode='bilinear', align_corners=False),
                F.interpolate(self.gt, scale_factor=1 / (scale * 2), mode='bilinear', align_corners=False),
                F.interpolate(self.gt, scale_factor=1 / scale, mode='bilinear', align_corners=False),
                F.interpolate(self.gt, scale_factor=1 / 5, mode='bilinear', align_corners=False)
            ]
            multiscale_loss = 0
            for i, gt_i in enumerate(gt_levels, start=1):
                if i < len(self.output):
                    multiscale_loss += self.cri_pix_multiscale(self.output[i], gt_i)
            l_total += multiscale_loss
            loss_dict['l_pix_multi'] = multiscale_loss

        # --------------------
        # Backpropagation
        # --------------------
        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # Exponential Moving Average (optional)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
