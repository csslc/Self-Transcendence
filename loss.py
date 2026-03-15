import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                # print(z_j.shape)
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss


class SILoss_ori:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
       

        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)


        return denoising_loss



class vae_diffloss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            accelerator=None, 
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, t_range=None, model_kwargs=None):
       

        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        if t_range is not None:
            a, b = t_range
            t_mask = (time_input >= a) & (time_input <= b)  # shape: [B,]
            t_mask = t_mask.view(-1, *([1] * (images.dim() - 1)))  # broadcast shape
        else:
            t_mask = None

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_output, model_output_fea  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # 1
        denoising_loss_fea = mean_flat((model_output_fea - model_target) ** 2)

        if t_mask is not None:
            denoising_loss_fea = denoising_loss_fea * t_mask

        return denoising_loss, denoising_loss_fea




class selftrans_Loss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform", 
            accelerator=None, 
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, guided_model, model, images, t_range=None, labels=None, tea_depth=None, stu_depth=None, cfg_scale=None, model_kwargs=None):
       

        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        if t_range is not None:
            a, b = t_range
            t_mask = (time_input >= a) & (time_input <= b)  # shape: [B,]
            t_mask = t_mask.view(-1, *([1] * (images.dim() - 1)))  # broadcast shape
        else:
            t_mask = None

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction


        model_output, model_output_fea  = model(model_input, time_input.flatten(), **model_kwargs, return_proj_fea=stu_depth)
        
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        
        with torch.no_grad():
            local_batch_size = model_input.shape[0]
            labels_uncond = 1000 * torch.ones(len(labels), dtype=labels.dtype, device=labels.device)
            labels_cfg = torch.cat([labels, labels_uncond], 0)
            model_input_cfg = torch.cat([model_input, model_input], 0)
            time_input_cfg = torch.cat([time_input, time_input], 0)
            model_kwargs_cfg = dict(y=labels_cfg)
            out_fea  = guided_model(model_input_cfg, time_input_cfg.flatten(), **model_kwargs_cfg, return_fea=tea_depth)
            fea_cond, fea_uncond = out_fea[0:local_batch_size], out_fea[local_batch_size:]
            fea_cfg = fea_uncond + cfg_scale * (fea_cond - fea_uncond)
            
        guided_loss = mean_flat((model_output_fea - fea_cfg) ** 2)


        if t_mask is not None:
            guided_loss = guided_loss * t_mask

        return denoising_loss, guided_loss




class selftrans_t2i_Loss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform", 
            accelerator=None, 
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, guided_model, model, images, t_range=None, context=None, y_null=None, tea_depth=None, stu_depth=None, cfg_scale=None, model_kwargs=None):
       

        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        if t_range is not None:
            a, b = t_range
            t_mask = (time_input >= a) & (time_input <= b)  # shape: [B,]
            t_mask = t_mask.view(-1, *([1] * (images.dim() - 1)))  # broadcast shape
        else:
            t_mask = None

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction


        model_output, model_output_fea  = model(model_input, time_input.flatten(), **model_kwargs, return_proj_fea=stu_depth)
        
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        
        with torch.no_grad():
            local_batch_size = model_input.shape[0]
            y_null = y_null * torch.ones(context.shape, dtype=context.dtype, device=context.device)
            contex_cfg = torch.cat([context, y_null], 0)
            model_input_cfg = torch.cat([model_input, model_input], 0)
            time_input_cfg = torch.cat([time_input, time_input], 0)
            model_kwargs_cfg = dict(context=contex_cfg)
            out_fea = guided_model(model_input_cfg, time_input_cfg.flatten(), **model_kwargs_cfg, return_fea=tea_depth)
            fea_cond, fea_uncond = out_fea[0:local_batch_size], out_fea[local_batch_size:]
            fea_cfg = fea_uncond + cfg_scale * (fea_cond - fea_uncond)
            
        guided_loss = mean_flat((model_output_fea - fea_cfg) ** 2)

        if t_mask is not None:
            guided_loss = guided_loss * t_mask

        return denoising_loss, guided_loss
