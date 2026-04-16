from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
from trellis.utils.visualize import visualize_coords


def generate_output_dir_name(prompt1: str, prompt2: str, base_dir: str = "outputs", case: int = 1) -> str:
    """
    Generate output directory name based on current time and two prompts.
    
    Args:
        prompt1: First prompt text
        prompt2: Second prompt text
        base_dir: Base directory path (default: "outputs")
        case: Case number (default: 1)
    
    Returns:
        str: Directory name in format: base_dir/YYYYMMDD_HHMMSS_prompt1&prompt2&case{case}
    """

    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    
    # Clean prompts: remove special characters, keep alphanumeric and spaces
    def clean_prompt(prompt: str) -> str:
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)
        cleaned = cleaned.replace(' ', '_')
        return cleaned.lower() if cleaned else "prompt"
    
    prompt1_clean = clean_prompt(prompt1)
    prompt2_clean = clean_prompt(prompt2)
    
    dir_name = f"{time_str}_{prompt1_clean}&{prompt2_clean}&case{case}"
    return os.path.join(base_dir, dir_name)

class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])

        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    def rotate_coords(self, coords: torch.Tensor, axis: str = 'x', k: int = 1, resolution: int = 64) -> torch.Tensor:
        """
        Rotate voxel coordinates by 90 degrees around specified axis.
        
        Args:
            coords: Tensor of shape [N, 4] where format is [batch, z, y, x]
            axis: Rotation axis ('x', 'y', or 'z')
            k: Number of 90-degree rotations (1, 2, or 3)
            resolution: Resolution of the voxel grid
            
        Returns:
            rotated_coords: Rotated coordinates with same shape as input
        """
        if coords.shape[0] == 0:
            return coords
        
        # Convert coords to occupancy grid for rotation
        occ_grid = torch.zeros(1, 1, resolution, resolution, resolution, 
                              dtype=torch.float32, device=coords.device)
        
        batch_indices = coords[:, 0].long()
        z_indices = coords[:, 1].long()
        y_indices = coords[:, 2].long()
        x_indices = coords[:, 3].long()
        
        # Filter valid indices
        valid_mask = (
            (batch_indices >= 0) & (batch_indices < 1) &
            (z_indices >= 0) & (z_indices < resolution) &
            (y_indices >= 0) & (y_indices < resolution) &
            (x_indices >= 0) & (x_indices < resolution)
        )
        
        if valid_mask.sum() > 0:
            occ_grid[batch_indices[valid_mask], 0, 
                    z_indices[valid_mask], y_indices[valid_mask], x_indices[valid_mask]] = 1.0
        
        # Rotate occupancy grid
        # occ_grid shape: [B, C, D, H, W] = [1, 1, Z, Y, X]
        if axis == 'x':
            # Rotate around X axis: rotate in Y-Z plane
            # dims=[2, 3] corresponds to [Z, Y] dimensions
            occ_rotated = torch.rot90(occ_grid, k=k, dims=[2, 3])
        elif axis == 'y':
            # Rotate around Y axis: rotate in Z-X plane
            # dims=[2, 4] corresponds to [Z, X] dimensions
            occ_rotated = torch.rot90(occ_grid, k=k, dims=[2, 4])
        elif axis == 'z':
            # Rotate around Z axis: rotate in Y-X plane
            # dims=[3, 4] corresponds to [Y, X] dimensions
            occ_rotated = torch.rot90(occ_grid, k=k, dims=[3, 4])
        else:
            raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', or 'z'")
        
        # Extract rotated coords
        rotated_coords = torch.argwhere(occ_rotated[0, 0] > 0.5)[:, [0, 1, 2]].int()  # [N, 3] z, y, x
        if rotated_coords.shape[0] > 0:
            # Add batch dimension
            rotated_coords = torch.cat([
                torch.zeros(rotated_coords.shape[0], 1, dtype=torch.int32, device=rotated_coords.device),
                rotated_coords
            ], dim=1)  # [N, 4] [batch, z, y, x]
        else:
            rotated_coords = torch.empty((0, 4), dtype=torch.int32, device=coords.device)
        
        return rotated_coords

    def sample_sparse_structure(
        self,
        cond1: dict,
        cond2: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        mode: str = 'dual',
        angle_check: bool = False,
        rotation_step: int = 0,
        rotation_info: dict = None,
        output_dir: Optional[str] = None,
        guidance: str = 'false',
        case: int = 1,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond1 (dict): The first conditioning information.
            cond2 (dict): The second conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            mode (str): Sampling mode ('dual' or 'single').
            angle_check (bool): Whether to check angles.
            rotation_step (int): Rotation step for dual mode.
            output_dir (str, optional): Output directory for saving results.
            guidance (str): Guidance method: 'false' (no guidance), 'noise_guidance' (use guided_structure_weight), 'space_control' (use t0_idx_value).
            case (int): Splitting case. case=1: obj1 upper half + obj2 lower half; case=2: obj1 lower half + obj2 lower half (rotated 180°).
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        # Add output_dir to sampler_params so it can be passed through kwargs
        if output_dir is not None:
            sampler_params['output_dir'] = output_dir

        if mode == 'dual':
            print("Dual mode")
            
            # Initialize variables for guidance
            input_noise = noise
            z_encoded = None
            control = None
            t0_idx_value = None
            
            # Only generate guidance structure if not angle_check and guidance is enabled
            if not angle_check and guidance != 'false':
                # Remove parameters that should not be passed to the model
                sampler_params_clean_early = {k: v for k, v in sampler_params.items() if k not in ["guided_structure_weight", "guided_noise_scale", "t0_idx_value"]}
                
                print("generating noise - 1")
                z_s_1 = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    cond1['cond'],
                    cond1['neg_cond'],
                    None,
                    None,
                    **sampler_params_clean_early,
                    verbose=True,
                ).samples

                print("generating noise - 2")
                z_s_2 = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    cond2['cond'],
                    cond2['neg_cond'],
                    None,
                    None,
                    **sampler_params_clean_early,
                    verbose=True,
                ).samples

                # Decode occupancy latent
                decoder = self.models['sparse_structure_decoder']
                coords_1 = torch.argwhere(decoder(z_s_1)>0)[:, [0, 2, 3, 4]].int()
                coords_2 = torch.argwhere(decoder(z_s_2)>0)[:, [0, 2, 3, 4]].int()

                # Use output_dir if provided, otherwise use default
                if output_dir is None:
                    output_dir = "outputs/default"
                os.makedirs(output_dir, exist_ok=True)

                # Split and merge voxels based on case parameter
                print(f"Splitting and merging voxels (case={case})...")
                resolution = 64
                half_point = resolution // 2  # 32

                if case == 1:
                    # Case 1: Janus split at y = half_point — stream1 y < half, stream2 y >= half.
                    # coords format: [batch, z, y, x]; index 2 is y.
                    coords_1_part = coords_1[coords_1[:, 2] < half_point]
                    coords_2_part = coords_2[coords_2[:, 2] >= half_point]
                    case_description = f"{coords_1_part.shape[0]} from coords_1 (y<half) + {coords_2_part.shape[0]} from coords_2 (y>=half)"
                    
                elif case == 2:
                    # Case 2: both streams use y < half_point; rotate obj2 half 180° (rot90 k=2, axis x) before merge.
                    coords_1_part = coords_1[coords_1[:, 2] < half_point]
                    coords_2_lower = coords_2[coords_2[:, 2] < half_point]
                    print("Rotating coords_2 lower half by 180° (rot90 k=2, axis='x')...")
                    coords_2_part = self.rotate_coords(coords_2_lower, axis='x', k=2, resolution=resolution)
                    case_description = f"{coords_1_part.shape[0]} from coords_1 (lower) + {coords_2_part.shape[0]} from coords_2 (lower, rotated 180°)"
                else:
                    raise ValueError(f"Unknown case: {case}. Use case=1 or case=2")

                # Merge the two parts
                coords_combined = torch.cat([coords_1_part, coords_2_part], dim=0)

                print(f"Merged voxel layout: {case_description} = {coords_combined.shape[0]} total")

                # ======================Encode combined voxel to latent======================
                # Convert combined coords to occupancy grid and encode to latent
                print("Encoding combined voxel to latent...")
                occ_grid = torch.zeros(num_samples, 1, resolution, resolution, resolution, 
                                      dtype=torch.float32, device=coords_combined.device)
                if coords_combined.shape[0] > 0:
                    batch_indices = coords_combined[:, 0].long()
                    z_indices = coords_combined[:, 1].long()
                    y_indices = coords_combined[:, 2].long()
                    x_indices = coords_combined[:, 3].long()

                    # Filter valid indices
                    valid_mask = (
                        (batch_indices >= 0) & (batch_indices < num_samples) &
                        (z_indices >= 0) & (z_indices < resolution) &
                        (y_indices >= 0) & (y_indices < resolution) &
                        (x_indices >= 0) & (x_indices < resolution)
                    )

                    if valid_mask.sum() > 0:
                        occ_grid[batch_indices[valid_mask], 0, 
                                z_indices[valid_mask], y_indices[valid_mask], x_indices[valid_mask]] = 1.0

                # Load encoder and encode to latent
                encoder_ckpt = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
                if 'sparse_structure_encoder' in self.models:
                    encoder = self.models['sparse_structure_encoder']
                else:
                    import trellis.models as models
                    print(f"Loading encoder from {encoder_ckpt}...")
                    encoder = models.from_pretrained(encoder_ckpt).eval().to(self.device)

                with torch.no_grad():
                    z_encoded = encoder(occ_grid, sample_posterior=False)

                # Check dimensions and adjust if needed
                if z_encoded.shape != noise.shape:
                    print(f"Warning: Encoded shape {z_encoded.shape} != noise shape {noise.shape}")
                    if z_encoded.shape[1] == flow_model.in_channels:
                        # Upsample spatial dimensions if needed
                        z_encoded = torch.nn.functional.interpolate(
                            z_encoded,
                            size=(reso, reso, reso),
                            mode='trilinear',
                            align_corners=False
                        )
                    else:
                        print("Channel mismatch, using original noise")
                        z_encoded = noise

                # Apply guidance method
                if guidance == 'noise_guidance':
                    # NOISE GUIDANCE: Blend encoded structure with random noise
                    structure_weight = sampler_params.get('guided_structure_weight', 0.4)
                    noise_scale = sampler_params.get('guided_noise_scale', 1.0)
                    
                    random_noise = torch.randn_like(z_encoded) * noise_scale
                    input_noise = structure_weight * z_encoded + (1 - structure_weight) * random_noise
                    print(f"Using NOISE GUIDANCE: structure_weight={structure_weight}, noise_scale={noise_scale}")
                    
                elif guidance == 'space_control':
                    # SPACE CONTROL: Use encoded structure as control with t0_idx_value
                    control = z_encoded
                    t0_idx_value = sampler_params.get('t0_idx_value', 10)
                    input_noise = noise  # Use random noise as starting point
                    print(f"Using SPACE CONTROL: t0_idx_value={t0_idx_value}")
                else:
                    # guidance == 'false': Use random noise
                    input_noise = noise
                    print("Using random noise (no guidance)")
            
            # Remove guidance-specific parameters from sampler_params to avoid passing them to model
            sampler_params_clean = {k: v for k, v in sampler_params.items() if k not in ["t0_idx_value", "guided_structure_weight", "guided_noise_scale"]}
            
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                input_noise,
                cond1['cond'],
                cond1['neg_cond'],
                cond2['cond'],
                cond2['neg_cond'],
                decoder=self.models['sparse_structure_decoder'],
                **sampler_params_clean,
                verbose=True,
                isSlat=False,
                angle_check=angle_check,
                rotation_step=rotation_step,
                rotation_info=rotation_info,
                case=case,
                control=control,
                t0_idx_value=t0_idx_value,
            )

            z_s_1 = z_s.samples_1
            z_s_2 = z_s.samples_2     

            # Decode occupancy latent
            decoder = self.models['sparse_structure_decoder']
            
            if not angle_check:
                coords_1 = torch.argwhere(decoder(z_s_1)>0)[:, [0, 2, 3, 4]].int()
                coords_2 = torch.argwhere(decoder(z_s_2)>0)[:, [0, 2, 3, 4]].int()

                return coords_1, coords_2
            elif angle_check:
                occ_1 = decoder(z_s_1)
                occ_2 = decoder(z_s_2)
                
                coords_1_z = torch.argwhere(occ_1>0)[:, [0, 2, 3, 4]].int()
                
                occ_2_rotated_x_1 = torch.rot90(occ_2, k=1, dims=[3, 4])
                occ_2_rotated_x_2 = torch.rot90(occ_2, k=2, dims=[3, 4])
                occ_2_rotated_x_3 = torch.rot90(occ_2, k=3, dims=[3, 4])
                
                occ_2_rotated_y_1 = torch.rot90(occ_2, k=1, dims=[2, 4])
                occ_2_rotated_y_2 = torch.rot90(occ_2, k=2, dims=[2, 4])
                occ_2_rotated_y_3 = torch.rot90(occ_2, k=3, dims=[2, 4])
                
                coords_2_x_1 = torch.argwhere(occ_2_rotated_x_1>0)[:, [0, 2, 3, 4]].int()
                coords_2_x_2 = torch.argwhere(occ_2_rotated_x_2>0)[:, [0, 2, 3, 4]].int()
                coords_2_x_3 = torch.argwhere(occ_2_rotated_x_3>0)[:, [0, 2, 3, 4]].int()
                
                coords_2_y_1 = torch.argwhere(occ_2_rotated_y_1>0)[:, [0, 2, 3, 4]].int()
                coords_2_y_2 = torch.argwhere(occ_2_rotated_y_2>0)[:, [0, 2, 3, 4]].int()
                coords_2_y_3 = torch.argwhere(occ_2_rotated_y_3>0)[:, [0, 2, 3, 4]].int()
                
                coords_2_z = torch.argwhere(occ_2>0)[:, [0, 2, 3, 4]].int()
                
                return coords_1_z, coords_2_x_1, coords_2_x_2, coords_2_x_3, coords_2_y_1, coords_2_y_2, coords_2_y_3, coords_2_z
        
        elif mode == 'single':
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                cond1['cond'],
                cond1['neg_cond'],
                None,
                None,
                **sampler_params,
                verbose=True,
            ).samples
        
            # Decode occupancy latent
            decoder = self.models['sparse_structure_decoder']
            coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

            return coords
        
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond1: dict = None,
        cond2: dict = None,
        coords: torch.Tensor = None,
        sampler_params: dict = {},
        mode: str = 'dual',
        feats: torch.Tensor = None,
        slat: str = '1'
        ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        if feats is None:
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
            )
        else:
            noise = sp.SparseTensor(
                feats=feats,
                coords=coords,
            )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        if mode == 'dual':
            
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                cond1['cond'],
                cond1['neg_cond'],
                cond2['cond'],
                cond2['neg_cond'],
                **sampler_params,
                verbose=True,
                isSlat=True
            ).samples

            std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
            slat = slat * std + mean

            return slat
        elif mode == 'single':
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                cond1['cond'],
                cond1['neg_cond'],
                None,
                None,
                **sampler_params,
                verbose=True,
            ).samples
            
            std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
            slat = slat * std + mean

            return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str =  None,
        prompt1: str = None,
        prompt2: str = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        angle_check: bool = False,
        rotation_step: int = 0,
        rotation_info: dict = None,
        output_dir: Optional[str] = None,
        guidance: str = 'false',
        case: int = 1,
        t0_idx_value: int = 10,
        guided_structure_weight: float = 0.3,
        blend_strategy: str = 'sdf_avg',
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            prompt1 (str): First text prompt for dual mode.
            prompt2 (str): Second text prompt for dual mode.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            angle_check (bool): Whether to check angles.
            rotation_step (int): Rotation step for dual mode.
            output_dir (str, optional): Output directory. If None, will be auto-generated based on time and prompts.
            guidance (str): Guidance method: 'false' (no guidance), 'noise_guidance' (use guided_structure_weight), 'space_control' (use t0_idx_value). Default is 'false'.
            case (int): Splitting case. case=1: obj1 upper half + obj2 lower half; case=2: obj1 lower half + obj2 lower half (rotated 180°). Default is 1.
            t0_idx_value (int): Time index for SPACE CONTROL τ₀. Default is 10.
            guided_structure_weight (float): Weight for guided structure (0.0-1.0). Default is 0.4.
            blend_strategy (str): Blend strategy for voxel blending. Default is 'sdf_avg'.
        """
        torch.manual_seed(seed)
        sparse_structure_sampler_params['t0_idx_value'] = t0_idx_value
        sparse_structure_sampler_params['guided_structure_weight'] = guided_structure_weight
        sparse_structure_sampler_params['blend_strategy'] = blend_strategy
        
        if prompt1 is not None and prompt2 is not None:
            cond1 = self.get_cond([prompt1])
            cond2 = self.get_cond([prompt2])
            
            if output_dir is not None:
                pass
            else:
                output_dir = generate_output_dir_name(prompt1, prompt2)

            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Output directory: {output_dir}")
            
            if not angle_check:

                coords_1, coords_2 = self.sample_sparse_structure(
                                        cond1, 
                                        cond2, 
                                        num_samples, 
                                        sparse_structure_sampler_params, 
                                        mode='dual', 
                                        angle_check=angle_check, 
                                        rotation_step=rotation_step,
                                        rotation_info=rotation_info,
                                        output_dir=output_dir,
                                        guidance=guidance,
                                        case=case
                                    )
                print("=== Finish Structure ===")

                slat_1 = self.sample_slat(cond1, cond2, coords_1, slat_sampler_params, slat='1')
                slat_2 = self.sample_slat(cond1, cond2, coords_2, slat_sampler_params, slat='2')

                return self.decode_slat(slat_1, formats), self.decode_slat(slat_2, formats)
            
            elif angle_check:
                coords_1_z, \
                coords_2_x_1, coords_2_x_2, coords_2_x_3, \
                coords_2_y_1, coords_2_y_2, coords_2_y_3, \
                coords_2_z = self.sample_sparse_structure(
                                        cond1, 
                                        cond2, 
                                        num_samples, 
                                        sparse_structure_sampler_params, 
                                        mode='dual', 
                                        angle_check=angle_check, 
                                        rotation_step=rotation_step,
                                        output_dir=output_dir,
                                        guidance=guidance,
                                        case=case
                                    )
                print("=== Finish Structure ===")
                
                
                slat_1_z = self.sample_slat(cond1, cond1, coords_1_z, slat_sampler_params)
                
                flow_model = self.models['slat_flow_model']
                feats=torch.randn(coords_2_x_1.shape[0], flow_model.in_channels).to(self.device)
                
                slat_2_x_1 = self.sample_slat(cond1, cond2, coords_2_x_1, slat_sampler_params, feats=feats)
                slat_2_x_2 = self.sample_slat(cond1, cond2, coords_2_x_2, slat_sampler_params, feats=feats)
                slat_2_x_3 = self.sample_slat(cond1, cond2, coords_2_x_3, slat_sampler_params, feats=feats)
                slat_2_y_1 = self.sample_slat(cond1, cond2, coords_2_y_1, slat_sampler_params, feats=feats)
                slat_2_y_2 = self.sample_slat(cond1, cond2, coords_2_y_2, slat_sampler_params, feats=feats)
                slat_2_y_3 = self.sample_slat(cond1, cond2, coords_2_y_3, slat_sampler_params, feats=feats)
                slat_2_z = self.sample_slat(cond1, cond2, coords_2_z, slat_sampler_params, feats=feats)
                
                return self.decode_slat(slat_1_z, formats), self.decode_slat(slat_2_x_1, formats), self.decode_slat(slat_2_x_2, formats), self.decode_slat(slat_2_x_3, formats), self.decode_slat(slat_2_y_1, formats), self.decode_slat(slat_2_y_2, formats), self.decode_slat(slat_2_y_3, formats), self.decode_slat(slat_2_z, formats)


        else:
            cond = self.get_cond([prompt])
            coords = self.sample_sparse_structure(cond, None, num_samples, sparse_structure_sampler_params, mode='single')
            slat = self.sample_slat(cond, None, coords, slat_sampler_params, mode = 'single')
            return self.decode_slat(slat, formats)
            
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
