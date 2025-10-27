import os
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import pickle as pkl
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from basicsr.utils.dist_util import get_dist_info
from torch import distributed as dist


@MODEL_REGISTRY.register()
class ClimateSRModel(SRModel):
    """
    Simplified ClimateSRModel for precipitation-only downscaling.
    - Single variable (precipitation)
    - Elevation as auxiliary input
    - log1p normalization
    """

    def __init__(self, opt):
        super(ClimateSRModel, self).__init__(opt)
        self.common_stat = opt['datasets']['common']
        self.var_stats_dict = self.init_stat()

        # AMP
        self.amp_training = self.opt.get('fp16', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_training)
        logger = get_root_logger()
        logger.info(f'Enable AMP training: {self.amp_training}')

    # ------------------------
    # Initialize model & losses
    # ------------------------
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Using EMA with decay {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            self.model_ema(0)
            self.net_g_ema.eval()

        # define main loss
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        self.cri_pix_multiscale = None

        self.setup_optimizers()
        self.setup_schedulers()

    # ------------------------
    # Load variable statistics
    # ------------------------
    def init_stat(self):
        logger = get_root_logger()
        var_stats_dict = {}

        # Only precipitation and elevation stats
        all_var = self.common_stat['listofVar'].copy()
        all_var.extend(self.common_stat['varName_gt'].copy())
        all_var.append(self.common_stat['hgt_obs'])

        for varName in all_var:
            stat_file = os.path.join(
                self.common_stat['dirName_stat'],
                varName.replace("_cut", "").replace("_fix", "") + "_stat.pkl"
            )

            if os.path.isfile(stat_file):
                logger.info(f"Load stat from {stat_file}")
                with open(stat_file, "rb") as f:
                    var_stats_dict[varName] = pkl.load(f)
            else:
                logger.warning(f"⚠️ Stat file not found: {stat_file}")
                var_stats_dict[varName] = {'mean': 0.0, 'std': 1.0}

        return var_stats_dict

    # ------------------------
    # Optimizer setup
    # ------------------------
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            return torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            return torch.optim.AdamW(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} not supported.')

    # ------------------------
    # Forward + backward
    # ------------------------
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.output = self.net_g(self.lq)

        if isinstance(self.output, (list, tuple)):
            self.output = [v.float() for v in self.output]
            out_main = self.output[0]
        else:
            out_main = self.output.float()

        loss_dict = OrderedDict()
        l_total = 0.0

        if self.cri_pix:
            l_pix = self.cri_pix(out_main, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ------------------------
    # Data feeding
    # ------------------------
    def feed_data(self, data):
        """
        Expected keys from dataset:
        data = {
            'lq': tensor(B, 1, Hc, Wc),          # ERA5 precipitation (low-res)
            'hgt': tensor(B, 1, Hh, Wh),         # elevation (high-res)
            'gt': tensor(B, 1, Hh, Wh),          # CHIRPS precipitation (high-res)
            'info': [filename]
        }
        """
        lq = data['lq'].to(self.device)
        hgt = data['hgt'].to(self.device)
        self.lq = {'lq': lq, 'hgt': hgt}
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.info = data.get('info', ['None'])

    # ------------------------
    # Inference
    # ------------------------
    def test(self):
        if hasattr(self, 'net_g_ema'):
            net = self.net_g_ema
        else:
            net = self.net_g
        net.eval()
        with torch.no_grad():
            self.output = net(self.lq)
        net.train()

    # ------------------------
    # Distributed validation
    # ------------------------
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        rank, world_size = get_dist_info()

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics']}
            self._initialize_best_metric_results(dataset_name, self.metric_results)

        if use_pbar and rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')

        for didx in range(rank, len(dataset), world_size):
            val_data = dataset[didx]
            for key in val_data.keys():
                if key != 'info':
                    val_data[key] = val_data[key].unsqueeze(0)
            self.feed_data(val_data)
            self.test()

            # output, gt
            output = self.output
            target = self.gt

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_res = calculate_metric({'img': output, 'img2': target}, opt_)
                    self.metric_results[name] += metric_res[0].item()

            if rank == 0 and use_pbar:
                pbar.update(1)

        if use_pbar and rank == 0:
            pbar.close()

        if rank == 0 and with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= len(dataset)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


# =====================================================
# ClimateSRAddHGTModel: wrapper with elevation support
# =====================================================
@MODEL_REGISTRY.register()
class ClimateSRAddHGTModel(ClimateSRModel):
    """
    Handles input dictionary {'lq': tensor, 'hgt': tensor} during training/testing.
    """

    def feed_data(self, data):
        lq = data['lq'].to(self.device)           # ERA5-Land precipitation
        hgt = data['hgt'].to(self.device)         # elevation
        self.lq = {'lq': lq, 'hgt': hgt}

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)  # CHIRPS precipitation
        self.info = data.get('info', ['None'])
