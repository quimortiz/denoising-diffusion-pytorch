import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__
import matplotlib.pyplot as plt


from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, extract, ModelPrediction, has_int_squareroot, cycle, divisible_by, num_to_groups, exists, Dataset
    



class GaussianDiffusionMLPEncoder(Module):
    def __init__(
        self,
        model,
        *,
        vector_size ,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        embed_dim = 32,
        image_size=64,
        encoder = None,
        decoder = None,
        loss_in_image_space = False,
        fix_encoder_decoder=False,
        z_space_weight = .01,
        clip_denoised = True
):
        super().__init__()

        assert encoder is not None, 'encoder must be provided'
        assert decoder is not None, 'decoder must be provided'

        self.clip_denoised = clip_denoised

        self.z_space_weight = z_space_weight
        self.fix_encoder_decoder = fix_encoder_decoder
        

        self.loss_in_image_space = loss_in_image_space
        self.image_size=image_size

        # TODO: allow for fixed encoder and decoder -- but this happens in the trainer.
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = embed_dim


        self.model = model
        self.vector_size = vector_size
        self.self_condition = False


        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'


        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = self.clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)

        # decode to image
        ret = self.decoder(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        # (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size,self.vector_size), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, img_start = None):
        b, nx = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))
        # i 

        # noise = torch.randn((b, self.embed_dim), device = self.device)


        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        # offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        #
        # if offset_noise_strength > 0.:
        #     offset_noise = torch.randn(x_start.shape[:2], device = self.device)
        #     noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.loss_in_image_space:
            if self.objective == 'pred_x0':
                assert img_start is not None, 'image_start must be provided if loss is in image space'
                img_pred = self.decoder(model_out)
                loss = F.mse_loss(img_pred, img_start, reduction='none')
                loss = reduce(loss, 'b ... -> b', 'mean')
                loss = loss * extract(self.loss_weight, t, loss.shape)
                out =  loss.mean()
            elif self.objective == 'pred_noise':
                # get back the predicted start
                pred_noise = model_out
                x_start = self.predict_start_from_noise(x, t, pred_noise)
                img_pred = self.decoder(x_start)
                loss = F.mse_loss(img_pred, img_start, reduction='none')
                loss = reduce(loss, 'b ... -> b', 'mean')
                loss = loss * extract(self.loss_weight, t, loss.shape)
                out =  loss.mean()

            else:
                raise ValueError(f'unknown objective {self.objective}')
            
            if self.objective == 'pred_noise':
                target = noise
            elif self.objective == 'pred_x0':
                target = x_start
            elif self.objective == 'pred_v':
                v = self.predict_v(x_start, t, noise)
                target = v
            else:
                raise ValueError(f'unknown objective {self.objective}')

            loss = F.mse_loss(model_out, target, reduction = 'none')
            loss = reduce(loss, 'b ... -> b', 'mean')

            loss = loss * extract(self.loss_weight, t, loss.shape)
            error_in_z = self.z_space_weight * loss.mean()

            #print('error in z', error_in_z, 'error in image', out )
            out += error_in_z
            return out
        else: 
            if self.objective == 'pred_noise':
                target = noise
            elif self.objective == 'pred_x0':
                target = x_start
            elif self.objective == 'pred_v':
                v = self.predict_v(x_start, t, noise)
                target = v
            else:
                raise ValueError(f'unknown objective {self.objective}')

            loss = F.mse_loss(model_out, target, reduction = 'none')
            loss = reduce(loss, 'b ... -> b', 'mean')

            loss = loss * extract(self.loss_weight, t, loss.shape)
            return loss.mean()

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(x)
        # convert to low dim 
        z = self.encoder(img)
        assert 'img_start' not in kwargs, 'img_start should not be passed as a keyword argument'
        return self.p_losses(z, t, *args, **kwargs,img_start=img)

class TrainerMLPEncoder:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        dataset= None,
        z_weight = .001,
        recon_weight=1.,
        weight_diffusion=1.,
        fix_encoder_decoder=False,
        tb_writer = None
    ):
        super().__init__()

        self.tb_writer = tb_writer

        self.z_weight = z_weight
        self.recon_weight = recon_weight
        self.weight_diffusion = weight_diffusion
        self.out_info = {} # use this to read and write!

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        # self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels
        #
        # if not exists(convert_image_to):
        #     convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.vector_size = diffusion_model.vector_size
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.dataset = dataset
        if dataset is not None:
            self.ds = dataset

        else:
            self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        params = []

        if fix_encoder_decoder:
            params = diffusion_model.model.parameters()

        else:
            params = diffusion_model.parameters()
    

        self.opt = Adam(params, lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        out_info = self.out_info
        out_info['loss'] = []
        out_info['recon_loss'] = []
        out_info['z_loss'] = []
        out_info['diff_loss'] = []

        counter_writer = 0
        image_counter = 0


        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            #print(self.model.device)

            while self.step < self.train_num_steps:

                diffusion_loss = 0.
                recon_loss = 0.
                z_loss = 0.
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.dataset is not None:
                        data = next(self.dl)[0].to(device)
                    else:
                        data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss =  self.weight_diffusion *  loss / self.gradient_accumulate_every
                        # loss.backward()
                        # # Check gradients
                        # for name, param in self.model.named_parameters():
                        #     if param.grad is not None:
                        #         print(f"{name} gradient norm {param.grad.norm()}")
                        # import sys
                        # sys.exit()

                        # recon loss
                        z = self.model.encoder(data)
                        x_recon = self.model.decoder(z)
                        _recon_loss =   self.recon_weight * F.mse_loss(x_recon, data) / self.gradient_accumulate_every
                        _z_loss =  self.z_weight * F.mse_loss(z, torch.zeros_like(z)) / self.gradient_accumulate_every

                        loss += _recon_loss
                        loss += _z_loss

                        diffusion_loss += loss.item()
                        recon_loss += _recon_loss.item()
                        total_loss += loss.item()
                        z_loss += _z_loss.item()

                    self.accelerator.backward(loss)


                #if self.step % 100 == 0:
                
                pbar.set_description(f'diffusion: {diffusion_loss:.4f} recon: {recon_loss:.4f}  z {z_loss:4f} total loss {total_loss:.4f}')
                pbar.update(1)
                if self.step % 100 == 0:
                    counter_writer += 1
                    self.tb_writer.add_scalar('diffusion loss', diffusion_loss , counter_writer) 
                    self.tb_writer.add_scalar('recon loss', recon_loss, counter_writer )
                    self.tb_writer.add_scalar('z loss', z_loss / 100, counter_writer)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):

                        out_info['loss'].append(loss.item())
                        out_info['recon_loss'].append(recon_loss)
                        out_info['diff_loss'].append(diffusion_loss)
                        out_info['z_loss'].append(z_loss)


                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            # lets encode and decode some real data
                            #import pdb; pdb.set_trace()

                            z = self.model.encoder(data)
                            x_recon = self.model.decoder(z)

                            print("max z is", z.max() , "min z is", z.min())
                            print("average z norm is", z.norm(dim = 1).mean())

                        all_images = torch.cat(all_images_list, dim = 0)
                        recon_images = x_recon[:self.batch_size]

                        fout = str(self.results_folder / f'sample-{milestone}.png')
                        print("saving to", fout)
                        utils.save_image(all_images,fout , nrow = int(math.sqrt(self.num_samples)))
                        fout = str(self.results_folder / f'recon-{milestone}.png')
                        print('saving recon images to', fout)
                        utils.save_image(recon_images, fout, nrow = int(math.sqrt(self.num_samples)))  


                        #self.tb_writer.add_image('sample_images', all_images, image_counter)
                        #self.tb_writer.add_image('recon_images', recon_images, image_counter)

                        # Log images to TensorBoard
                        self.tb_writer.add_image('sample_images', utils.make_grid(all_images, nrow=int(math.sqrt(self.num_samples))), image_counter)
                        self.tb_writer.add_image('recon_images', utils.make_grid(recon_images, nrow=int(math.sqrt(self.num_samples))), image_counter)



                        image_counter += 1

                        # def plot_data(ax,X):
                        #     XX = X[:,0]
                        #     YY = X[:,1]
                        #     ax.plot(XX,YY, 'o')
                        #
                        #
                        # X = self.model.sample(batch_size = 128, return_all_timesteps = False)
                        #
                        #
                        # fig, ax = plt.subplots(1,1)
                        #
                        # plot_data(ax,X)
                        # ax.set_aspect('equal', 'box')
                        # ax.set_xlim(-1.5,1.5)
                        # ax.set_ylim(-1.5,1.5)
                        # # plt.show()
                        # fileout = str(self.results_folder / f'sample-{milestone}.png')
                        # print(f"saving to {fileout}")
                        # plt.savefig(fileout)
                        # plt.close()

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)


        accelerator.print('training complete')
