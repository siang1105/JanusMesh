from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import os
import open3d as o3d
import imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .ss_blend_denoise import blend_and_reencode_voxel_to_latent
# from test_encoder_decoder import test_pred_x0_encode_decode, compare_latent_distributions
import trellis.modules.sparse as sp
from trellis.utils import render_utils

class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['decoder_gs', 'encoder', 'decoder', 'slat_normalization', 'output_dir', 'control', 't0_idx_value', 'blend_strategy', 'guided_structure_weight', 'guided_noise_scale']}
        
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if isinstance(cond, dict):
            cond_tensor = cond['cond']
            if cond_tensor.shape[0] == 1 and x_t.shape[0] > 1:
                cond_tensor = cond_tensor.repeat(x_t.shape[0], 1)
        else:
            cond_tensor = cond

        return model(x_t, t, cond_tensor, **model_kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v
    
    def _get_model_prediction_blend_cond(
        self, 
        model, 
        x_t_1, 
        x_t_2, 
        t, 
        cond1, 
        cond2, 
        neg_cond1=None, 
        neg_cond2=None, 
        blend_weight=0.5, 
        step: Optional[int] = None, 
        use_voxel_blend=True, 
        angle_check: bool = False, 
        rotation_step: int = 0,
        rotation_info: dict = None,
        case: int = 1,
        **kwargs
    ):
        pred_v_1 = self._inference_model(model, x_t_1, t, cond1, neg_cond=neg_cond1, **kwargs)
        pred_v_2 = self._inference_model(model, x_t_2, t, cond2, neg_cond=neg_cond2, **kwargs)
        
        pred_x_0_1, pred_eps_1 = self._v_to_xstart_eps(x_t=x_t_1, t=t, v=pred_v_1)
        pred_x_0_2, pred_eps_2 = self._v_to_xstart_eps(x_t=x_t_2, t=t, v=pred_v_2)
        
        # Get output_dir from kwargs (may be passed as output_dir or save_dir)
        output_dir = kwargs.get('output_dir')
        
        # Get blend_strategy from kwargs, default to 'sdf_avg'
        blend_strategy = kwargs.get('blend_strategy', 'sdf_avg')
        
        pred_x_0_1_new, pred_x_0_2_new = blend_and_reencode_voxel_to_latent(
                    pred_x_0_1, pred_x_0_2, count=step,
                    blend_strategy=blend_strategy,
                    threshold=0.5,
                    blend_weight=0.5,
                    step=step,
                    angle_check=angle_check,
                    rotation_step=rotation_step,
                    rotation_info=rotation_info,
                    output_dir=output_dir,
                    case=case,
                )
        
        pred_x_0_1 = pred_x_0_1_new
        pred_x_0_2 = pred_x_0_2_new
        
        pred_v_1 = (x_t_1 - pred_x_0_1) / t
        pred_v_2 = (x_t_2 - pred_x_0_2) / t
        pred_eps_1 = self._xstart_to_eps(x_t_1, t, pred_x_0_1)
        pred_eps_2 = self._xstart_to_eps(x_t_2, t, pred_x_0_2)
        
        return (pred_x_0_1, pred_eps_1, pred_v_1), (pred_x_0_2, pred_eps_2, pred_v_2)

    def _get_model_prediction_blend_cond_slat(
        self, 
        model, 
        x_t,
        t, 
        cond1, 
        cond2, 
        neg_cond1=None, 
        neg_cond2=None, 
        blend_weight=0.5, 
        step: Optional[int] = None,
        **kwargs
        ):
            pred_v_1 = self._inference_model(model, x_t, t, cond1, neg_cond=neg_cond1, **kwargs)
            pred_v_2 = self._inference_model(model, x_t, t, cond2, neg_cond=neg_cond2, **kwargs)
            pred_x_0_1, pred_eps_1 = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_1)
            pred_x_0_2, pred_eps_2 = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_2)
            
            pred_v = blend_weight * pred_v_1 + (1 - blend_weight) * pred_v_2
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
            return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t_1,
        x_t_2,
        t: float,
        t_prev: float,
        cond1: Optional[Any] = None,
        cond2: Optional[Any] = None,
        neg_cond1: Optional[Any] = None,
        neg_cond2: Optional[Any] = None,
        blend_weight=0.5,
        step: Optional[int] = None,
        angle_check: bool = False,
        rotation_step: int = 0,
        rotation_info: dict = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder']}
        case = kwargs.pop('case', 1)  # Extract and remove 'case' from kwargs to avoid duplicate
        (pred_x_0_1, pred_eps_1, pred_v_1), (pred_x_0_2, pred_eps_2, pred_v_2) = \
            self._get_model_prediction_blend_cond(
                model, 
                x_t_1, 
                x_t_2, 
                t, 
                cond1, 
                cond2, 
                neg_cond1, 
                neg_cond2, 
                blend_weight, 
                step=step, 
                angle_check=angle_check, 
                rotation_step=rotation_step,
                rotation_info=rotation_info,
                case=case,
                **kwargs
            )

        pred_x_prev_1 = x_t_1 - (t - t_prev) * pred_v_1
        pred_x_prev_2 = x_t_2 - (t - t_prev) * pred_v_2
        
        return edict({
            "pred_x_prev_1": pred_x_prev_1, 
            "pred_x_0_1": pred_x_0_1,
            "pred_x_prev_2": pred_x_prev_2, 
            "pred_x_0_2": pred_x_0_2
        })

    @torch.no_grad()
    def sample_once_slat(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond1: Optional[Any] = None,
        cond2: Optional[Any] = None,
        neg_cond1: Optional[Any] = None,
        neg_cond2: Optional[Any] = None,
        blend_weight=0.5,
        step: Optional[int] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder']}
        
        pred_x_0, pred_eps, pred_v = self._get_model_prediction_blend_cond_slat(
        model, x_t, t, cond1, cond2, neg_cond1, neg_cond2, blend_weight, step=step, **kwargs)
        
        pred_x_prev = x_t - (t - t_prev) * pred_v
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond1: Optional[Any] = None,
        cond2: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        isSlat: bool = False,
        angle_check: bool = False,
        rotation_step: int = 0,
        rotation_info: dict = None,
        case: int = 1,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        if not isSlat:
            # SPACE CONTROL: anchor mixture at τ₀.
            if 'control' in kwargs and kwargs['control'] is not None:
                control = kwargs['control']
                t0_idx_value = kwargs.get('t0_idx_value', 10)
                t0 = t_seq[int(t0_idx_value)]
        
                # z_t0 = t0 * z1 + (1 - t0) * z_c0
                sample_1 = noise * t0 + control * (1 - t0)
                sample_2 = noise * t0 + control * (1 - t0)
                print(f"Applied SPACE CONTROL: t0_idx_value={t0_idx_value}, t0={t0:.4f}")
            else:
                sample_1 = noise
                sample_2 = noise
                t0 = None
            # ================================================

            ret = edict({
                "samples_1": None, "pred_x_t_1": [], "pred_x_0_1": [],
                "samples_2": None, "pred_x_t_2": [], "pred_x_0_2": []
            })
            for step, (t, t_prev) in tqdm(enumerate(t_pairs), desc="Sampling", disable=not verbose):
                # While t > t0, skip updates (freeze structure from τ₀).
                if 'control' in kwargs and kwargs['control'] is not None and t0 is not None and t > t0:
                    print(f"Skipping control at step {step} because t > t0")
                    continue  # No Euler step; keep current latents.
                # ================================================
                out = self.sample_once(
                    model, 
                    sample_1, 
                    sample_2, 
                    t, 
                    t_prev, 
                    cond1, 
                    cond2, 
                    step=step, 
                    angle_check=angle_check, 
                    rotation_step=rotation_step, 
                    rotation_info=rotation_info,
                    case=case,
                    **kwargs
                )

                sample_1 = out.pred_x_prev_1
                sample_2 = out.pred_x_prev_2

                ret.pred_x_t_1.append(out.pred_x_prev_1)
                ret.pred_x_0_1.append(out.pred_x_0_1)
                ret.pred_x_t_2.append(out.pred_x_prev_2)
                ret.pred_x_0_2.append(out.pred_x_0_2)
            if "decoder" in kwargs:
                decoder = kwargs["decoder"]
            ret.samples_1 = sample_1
            ret.samples_2 = sample_2
            return ret
        else:
            sample = noise

            ret = edict({
                "samples": None, "pred_x_t": [], "pred_x_0": []
            })

            for step, (t, t_prev) in tqdm(enumerate(t_pairs), desc="Sampling", disable=not verbose):
                out = self.sample_once_slat(
                    model, 
                    sample,
                    t, 
                    t_prev, 
                    cond1, 
                    cond2, 
                    step=step, 
                    **kwargs
                )

                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)

            ret.samples = sample
            return ret

    # ==================== Single mode ====================
    @torch.no_grad()
    def sample_once_single(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample_single(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once_single(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_slat_euler(
        self,
        model,
        noise,
        cond1: Optional[Any] = None,
        cond2: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)

class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond1,
        neg_cond1,
        cond2,
        neg_cond2,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        if cond2 is not None:
            return super().sample(model, noise, cond1, cond2, steps, rescale_t, verbose, neg_cond1=neg_cond1, neg_cond2=neg_cond2, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
        else:
            return super().sample_single(model, noise, cond1, steps, rescale_t, verbose, neg_cond=neg_cond1, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)