from abc import *
import os 
import torch
import json 
import os
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode

from synctweedies.utils.mesh_utils import *
from synctweedies.utils.image_utils import *
from synctweedies.renderer.mesh.mesh_renderer import UVProjection as UVP
from synctweedies.method_configs.case_config import (CANONICAL_DENOISING_ZT,
                                                INSTANCE_DENOISING_XT,
                                                JOINT_DENOISING_XT,
                                                JOINT_DENOISING_ZT, METHOD_MAP)
from synctweedies.model.base_model import BaseModel

from diffusers import ControlNetModel
try:
    # Newer diffusers versions
    from diffusers.utils.torch_utils import is_compiled_module
except ImportError:
    # Older diffusers versions
    from diffusers.utils import is_compiled_module


color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
            "red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
            "green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
            "aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
            "purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


class MeshTextureModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}")
        self.initialize()

        super().__init__()
        
        
    def initialize(self):
        super().initialize()
        
        self.mesh_path = os.path.abspath(self.config.mesh)
        self.result_dir = f"{self.output_dir}/results"
        self.intermediate_dir = f"{self.output_dir}/intermediate"

        dirs = [self.output_dir, self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "mesh_run_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)


    def init_mapper(self):
        self.camera_poses = []
        self.attention_mask=[]
        self.centers = camera_centers = ((0,0,0),)

        cam_count = len(self.config.camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(self.config.camera_azims):
            if azim < 0:
                azim += 360
            self.camera_poses.append((0, azim))
            self.attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
            if abs(azim) < front_view_diff:
                front_view_idx = i
                front_view_diff = abs(azim)
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)

        if self.config.top_cameras:
            self.camera_poses.append((30, 0))
            self.camera_poses.append((30, 180))

            self.attention_mask.append([front_view_idx, cam_count])
            self.attention_mask.append([back_view_idx, cam_count+1])

        ref_views = []
        if len(ref_views) == 0:
            ref_views = [front_view_idx]

        self.group_metas = split_groups(self.attention_mask, self.config.max_batch_size, ref_views)

        self.uvp = UVP(texture_size=self.config.latent_tex_size, render_size=self.config.latent_view_size, sampling_mode=self.config.rasterize_mode, channels=4, device=self.model._execution_device)
        if self.mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(self.mesh_path, scale_factor=self.config.mesh_scale or 1, autouv=self.config.mesh_autouv)
        elif self.mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(self.mesh_path, scale_factor=self.config.mesh_scale or 1, autouv=self.config.mesh_autouv)
        elif self.mesh_path.lower().endswith(".ply"):
            self.uvp.load_ply_mesh(self.mesh_path, scale_factor=self.config.mesh_scale or 1, autouv=self.config.mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=2.0)


        self.uvp_rgb = UVP(texture_size=self.config.rgb_tex_size, render_size=self.config.rgb_view_size, sampling_mode=self.config.rasterize_mode, channels=3, device=self.model._execution_device)
        self.uvp_rgb.mesh = self.uvp.mesh.clone()
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=2.0)
        _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False, disable=self.config.disable_voronoi)

        color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=self.model.text_encoder.dtype, device=self.model._execution_device)
        color_images = torch.ones(
            (1,1,self.config.latent_view_size*8, self.config.latent_view_size*8),
            device=self.model._execution_device, 
            dtype=self.model.text_encoder.dtype
        ) * color_images
        color_images *= ((0.5*color_images)+0.5)
        color_latents = encode_latents(self.model.vae, color_images)

        self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}
        print("Done Initialization")


    def forward_mapping(
        self, 
        z_t, 
        bg, 
        cur_index, 
        **kwargs,
    ):
        if z_t.dim() == 4:
            z_t = z_t.squeeze(0)
        self.uvp.set_texture_map(z_t)
        avg_x_t = self.uvp.render_textured_views()

        foregrounds = [view[:-1] for view in avg_x_t]
        masks = [view[-1:] for view in avg_x_t]
        t = self.model.scheduler.timesteps[cur_index]

        add_noise = kwargs.get("add_noise", False)
        if bg == None:
            bg_color = kwargs["background_colors"]
            background_latents = [self.color_latents[color] for color in bg_color]
            add_noise = True 
            
        else:
            background_latents = [prev_latent for prev_latent in bg]
            if cur_index == 0:
                t = t+1
        
        composited_tensor = composite_rendered_view(scheduler=self.model.scheduler, 
                                                    backgrounds=background_latents, 
                                                    foregrounds=foregrounds, 
                                                    masks=masks, 
                                                    t=t,
                                                    add_noise=add_noise)
            
        latents = composited_tensor.type(background_latents[0].dtype)

        return latents


    def inverse_mapping(
        self, 
        x_t, 
        main_views, 
        exp, 
        **kwargs
    ):
        
        views = [view.to(self.uvp.device) for view in x_t] 
        rendered_views, z_t, visibility_weights = self.uvp.bake_texture(views=views, 
                                                                  main_views=main_views, 
                                                                  exp=exp,
                                                                  disable=self.config.disable_voronoi)

        return z_t


    def compute_noise_preds(
        self, 
        xts, 
        timestep, 
        **kwargs
    ):
     
        return self.model.compute_noise_preds(xts, timestep, **kwargs)


    @torch.no_grad()
    def __call__(self):
        num_timesteps = self.model.scheduler.config.num_train_timesteps
        initial_controlnet_conditioning_scale = self.config.conditioning_scale
        controlnet_conditioning_end_scale = self.config.conditioning_scale_end
        control_guidance_end = self.config.control_guidance_end
        control_guidance_start = self.config.control_guidance_start
        log_interval = self.config.log_interval
        view_fast_preview = self.config.view_fast_preview
        tex_fast_preview = self.config.tex_fast_preview
        ref_attention_end = self.config.ref_attention_end
        guidance_rescale = self.config.guidance_rescale
        multiview_diffusion_end = self.config.mvd_end
        shuffle_background_change = self.config.shuffle_bg_change
        shuffle_background_end = self.config.shuffle_bg_end
        callback = None

        controlnet = self.model.controlnet._orig_mod if is_compiled_module(self.model.controlnet) else self.model.controlnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 0. Default height and width to unet
        height = self.config.latent_view_size*8
        width = self.config.latent_view_size*8
        height = height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = width or self.model.unet.config.sample_size * self.model.vae_scale_factor

        controlnet_conditioning_scale = float(self.config.conditioning_scale)
        # prompt = f"Best quality, extremely detailed {self.config.prompt}"
        prompt = f"Best quality, {self.config.prompt}"
        negative_prompt = self.config.negative_prompt
        callback_steps = 1

        if self.config.sdedit:
            assert self.config.sdedit_prompt != None, f"Provide sdedit_prompt: {self.config.sdedit_prompt}"
            print("Running SDEdit from ", prompt, " --> ", self.config.sdedit_prompt)
            prompt = self.config.sdedit_prompt

        # 1. Check inputs. Raise error if not correct
        self.model.check_inputs(
            prompt=prompt,
            image=torch.zeros((1, 3, height, width), device=self.model._execution_device),
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        # 2. Define call parameters
        num_images_per_prompt = 1
        if prompt is not None and isinstance(prompt, list):
            assert len(prompt) == 1 and len(negative_prompt) == 1, "Only implemented for 1 (negative) prompt"  
        assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
        batch_size = len(self.uvp.cameras)

        device = self.model._execution_device
        guidance_scale = self.config.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = self.config.guess_mode or global_pool_conditions

        prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

        cross_attention_kwargs = None
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self.model._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )

        negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
        prompt_embed_dict = dict(zip(direction_names, [emb for emb in prompt_embeds]))
        negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))

        secondary_prompt_text = getattr(self.config, "secondary_prompt", None)
        secondary_negative_prompt_text = getattr(self.config, "secondary_negative_prompt", None) or self.config.negative_prompt
        secondary_prompt_embed_dict = prompt_embed_dict
        secondary_negative_prompt_embed_dict = negative_prompt_embed_dict

        if secondary_prompt_text:
            secondary_prompt = f"Best quality, {secondary_prompt_text}"
            secondary_prompt_dir, secondary_negative_dir = prepare_directional_prompt(secondary_prompt, secondary_negative_prompt_text)

            secondary_prompt_embeds = self.model._encode_prompt(
                secondary_prompt_dir,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                secondary_negative_dir,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
            )
            secondary_negative_prompt_embeds, secondary_prompt_embeds = torch.chunk(secondary_prompt_embeds, 2)
            secondary_prompt_embed_dict = dict(zip(direction_names, [emb for emb in secondary_prompt_embeds]))
            secondary_negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in secondary_negative_prompt_embeds]))
            
        tertiary_prompt_text = getattr(self.config, "tertiary_prompt", None)
        tertiary_negative_prompt_text = getattr(self.config, "tertiary_negative_prompt", None) or self.config.negative_prompt
        tertiary_prompt_embed_dict = prompt_embed_dict
        tertiary_negative_prompt_embed_dict = negative_prompt_embed_dict

        if tertiary_prompt_text:
            tertiary_prompt = f"Best quality, {tertiary_prompt_text}"
            tertiary_prompt_dir, tertiary_negative_dir = prepare_directional_prompt(tertiary_prompt, tertiary_negative_prompt_text)

            tertiary_prompt_embeds = self.model._encode_prompt(
                tertiary_prompt_dir,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                tertiary_negative_dir,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
            )
            tertiary_negative_prompt_embeds, tertiary_prompt_embeds = torch.chunk(tertiary_prompt_embeds, 2)
            tertiary_prompt_embed_dict = dict(zip(direction_names, [emb for emb in tertiary_prompt_embeds]))
            tertiary_negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in tertiary_negative_prompt_embeds]))
            

        self.uvp.to(self.model._execution_device)

        cond_type = self.config.cond_type
        conditioning_images, rgb_masks = get_conditioning_images(self.uvp, height, cond_type=cond_type)
        conditioning_images = conditioning_images.type(prompt_embeds.dtype)
        cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
        cond = np.concatenate([img for img in cond], axis=1)
        numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.png")

        num_inference_steps = self.config.steps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        num_channels_latents = self.model.unet.config.in_channels
        generator=torch.manual_seed(self.config.seed)

        case_name = METHOD_MAP[str(self.config.case_num)]
        z_t_given = None 
        if self.config.case_num in INSTANCE_DENOISING_XT or self.config.case_num in JOINT_DENOISING_XT:
            z_t_given = False 
        elif self.config.case_num in CANONICAL_DENOISING_ZT or self.config.case_num in JOINT_DENOISING_ZT:
            z_t_given = True
        else:
            raise NotImplementedError(f"{self.config.case_num}")

        # SDEdit 
        if self.config.sdedit:
            mesh_root = "/".join(self.config.mesh.split("/")[:-1])
            tex_path = os.path.join(mesh_root, "mesh_texture.png")
            assert os.path.exists(tex_path), f"SDEdit needs texture image {tex_path}"

            sdedit_timestep = self.config.sdedit_timestep
            print(f"Refinement : setting texture at timestep {sdedit_timestep}")
            sdedit_timestep_index = int(self.model.scheduler.num_inference_steps * sdedit_timestep)
            sdedit_start_time = self.model.scheduler.timesteps[sdedit_timestep_index]

            tex_resize = Resize((self.config.rgb_tex_size), interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
                
            # A) Set mesh with GT RGB texture map
            rgb_gt_tex = Image.open(tex_path).convert("RGB")
            rgb_gt_tex_resized = tex_resize(rgb_gt_tex)
            rgb_gt_tex_resized = pil_to_torch(rgb_gt_tex_resized).squeeze(0) # 3 1024 1024 
            self.uvp_rgb.set_texture_map(rgb_gt_tex_resized)

            # B) Render GT images and obtain Z_0
            latent_gt_img_list = []
            rgb_gt_img_list = []
            for _i in range(len(self.uvp_rgb.cameras)):
                rgb_gt_img = self.uvp_rgb.renderer(self.uvp_rgb.mesh.to(self.uvp_rgb.device), 
                                                cameras=self.uvp_rgb.cameras[_i], 
                                                lights=self.uvp_rgb.lights, 
                                                device=self.uvp_rgb.device)[..., :-1].half()

                rgb_gt_img = rgb_gt_img.permute(0, 3, 1, 2)
                rgb_gt_img_list.append(rgb_gt_img)
                log_rgb_gt_img = F.interpolate(rgb_gt_img, 
                                                size=(256, 256), 
                                                mode="nearest", 
                                                antialias=False)
                torch_to_pil(log_rgb_gt_img).save(f"{self.result_dir}/initial_views_rgb_{_i}.png")
                
                rgb_gt_img_resized = F.interpolate(rgb_gt_img, 
                                    size=(height, width), 
                                    mode="nearest", 
                                    antialias=False)

                latent_gt_img = encode_latents(self.model.vae, rgb_gt_img_resized)
                latent_gt_img_list.append(latent_gt_img)

            # C) Bake latent texture map using z_0 and add noise z_t
            temp_views = [view.squeeze(0) for view in latent_gt_img_list]
            _, latent_gt_texture, _ = self.uvp.bake_texture(views=temp_views, main_views=[], exp=0)

            tex_noise = torch.randn_like(latent_gt_texture)
            latent_tex = self.model.scheduler.add_noise(latent_gt_texture, tex_noise, sdedit_start_time) 
            assert latent_tex.dim() == 3 # (4, 1536, 1536)
            self.uvp.set_texture_map(latent_tex)

            # D) Add noise to latent image
            latents = torch.cat(latent_gt_img_list, axis=0)
            latent_noise = torch.randn_like(latents)
            latents = self.model.scheduler.add_noise(latents, latent_noise, sdedit_start_time)
            assert latents.dim() == 4

        else:
            latents = self.model.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            ) # (B, 4, 96, 96)
            latent_tex = self.uvp.set_noise_texture()

            if (self.config.case_num in INSTANCE_DENOISING_XT or self.config.case_num in JOINT_DENOISING_XT) and self.config.initialize_xt_from_zt:
                print(f"Initial z_T is given. x_ts are obtained from forward operation.")
                noise_views = self.uvp.render_textured_views()
                foregrounds = [view[:-1] for view in noise_views]
                masks = [view[-1:] for view in noise_views]
                composited_tensor = composite_rendered_view(
                    self.model.scheduler, 
                    latents, 
                    foregrounds, 
                    masks, 
                    timesteps[0]+1,
                    add_noise=True
                )
                latents = composited_tensor.type(latents.dtype)
            else:
                print(f"Initial z_T is NOT given. x_ts are individually initialized.")

        eta = 0.0
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)

        if self.config.sdedit:
            sdedit_timestep_index = int(self.model.scheduler.num_inference_steps * sdedit_timestep)
            timesteps = timesteps[sdedit_timestep_index:]

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        intermediate_results = []
        background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]

        main_views = []
        exp = 0
        alphas = self.model.scheduler.alphas_cumprod ** (0.5)
        sigmas = (1 - self.model.scheduler.alphas_cumprod) ** (0.5)
        
        # ========= Split view prompts by azimuth angle with smooth transition =========
        num_views = len(self.camera_poses)
        
        # Transition width in degrees (how wide the blending region is)
        # 0.0 means no transition (hard switch), > 0 means smooth blending
        transition_width = getattr(self.config, 'prompt_transition_width', 0.0)
        transition_extension = getattr(self.config, 'prompt_transition_extension', 2.0)
        extended_width = transition_extension * transition_width

        pos_embeds_per_view = []
        neg_embeds_per_view = []

        def blend_prompts(embed1, embed2, weight1):
            """Blend two prompt embeddings with given weight for embed1"""
            weight1 = torch.clamp(torch.tensor(weight1, device=embed1.device, dtype=embed1.dtype), 0.0, 1.0)
            weight2 = 1.0 - weight1
            return weight1 * embed1 + weight2 * embed2

        def compute_transition_weight(azim, boundary, transition_width, use_first_prompt_region):
            """
            Compute weight for first prompt based on distance from boundary.
            Args:
                azim: azimuth angle (0-360, normalized)
                boundary: boundary angle where transition happens
                transition_width: width of transition region (degrees)
                use_first_prompt_region: True if azim is in first prompt region (before boundary)
            Returns:
                weight for first prompt (1.0 = full first prompt, 0.0 = full second prompt)
            """
            # If no transition, return hard switch weight
            if transition_width <= 0:
                return 1.0 if use_first_prompt_region else 0.0
            
            # Compute distance to boundary (handle wrapping for boundaries like 270°)
            if boundary > 180:
                # For boundaries > 180, check both directions (e.g., 270°: check distance to 270° and to -90°)
                dist_to_boundary = min(abs(azim - boundary), abs(azim - (boundary - 360)))
            else:
                dist_to_boundary = abs(azim - boundary)
            
            # Weight calculation for smooth transition with extended blending range:
            # Use a smooth decay function that allows blending even beyond transition_width
            # The effective blending range is extended to allow more angles to participate
            # 
            # Use a smooth sigmoid-like function: weight = 1 / (1 + exp(k * (dist - transition_width)))
            # Or use a simpler smooth linear decay with extended range
            #
            # Extended transition range: use transition_extension * transition_width as the effective blending range
            # This allows angles further from boundary to still have some blending
            # Note: extended_width is now defined in outer scope
            
            if use_first_prompt_region:
                # In first prompt region, weight decreases from 1.0 (far) to 0.5 (at boundary)
                # Use linear interpolation with transition_width as the main blending zone
                # Beyond transition_width, use exponential decay for smoother falloff
                if dist_to_boundary <= transition_width:
                    # Within transition_width: linear from 1.0 to 0.5
                    raw_weight = 0.5 + 0.5 * (dist_to_boundary / transition_width)
                else:
                    # Beyond transition_width: exponential decay to maintain smoothness
                    # weight = 0.5 + 0.5 * exp(-k * (dist - transition_width))
                    # where k is chosen so that at extended_width, weight is close to 1.0
                    excess_dist = dist_to_boundary - transition_width
                    decay_range = extended_width - transition_width
                    if decay_range > 0:
                        # Exponential decay: at extended_width, weight should be ~0.95
                        # 0.5 + 0.5 * exp(-k * decay_range) = 0.95
                        # exp(-k * decay_range) = 0.9
                        # k = -ln(0.9) / decay_range
                        k = -np.log(0.9) / decay_range
                        decay_factor = np.exp(-k * excess_dist)
                        raw_weight = 0.5 + 0.5 * decay_factor
                    else:
                        raw_weight = 1.0
            else:
                # In second prompt region, weight increases from 0.0 (far) to 0.5 (at boundary)
                if dist_to_boundary <= transition_width:
                    # Within transition_width: linear from 0.0 to 0.5
                    raw_weight = 0.5 * (1.0 - dist_to_boundary / transition_width)
                else:
                    # Beyond transition_width: exponential decay
                    excess_dist = dist_to_boundary - transition_width
                    decay_range = extended_width - transition_width
                    if decay_range > 0:
                        k = -np.log(0.9) / decay_range
                        decay_factor = np.exp(-k * excess_dist)
                        raw_weight = 0.5 * decay_factor
                    else:
                        raw_weight = 0.0
            
            # Clamp to [0, 1] but allow slight extension for very smooth blending
            weight = max(0.0, min(1.0, raw_weight))
            
            print(f"  compute_transition_weight: azim={azim:.1f}, boundary={boundary}, dist={dist_to_boundary:.2f}, transition_width={transition_width:.2f}, use_first={use_first_prompt_region}, raw_weight={raw_weight:.3f}, final_weight={weight:.3f}")
                
            return weight

        for i, pose in enumerate(self.camera_poses):
            elev, azim = pose  # Azimuth in [0, 360] (normalized in init_mapper).

            # Special handling for 'third' mode with three prompts
            if self.config.azim_split_mode == 'third':
                # Sectors: prompt1 on (300°, 360] U [0°, 60°]; prompt2 on (60°, 180°]; prompt3 on (180°, 300°].
                # Normalize azim to 0-360 range if needed (should already be normalized, but just in case)
                azim_normalized = azim if azim >= 0 else azim + 360
                
                # Boundaries at 60°, 180°, 300°
                boundaries = [60, 180, 300]
                
                # Determine which region and compute weights
                if azim_normalized > 300 or azim_normalized <= 60:
                    # Region 1: 300° → 360° and 0° → 60°
                    # Check if near boundary at 60° or 300°
                    if azim_normalized <= 60:
                        # Near 60° boundary, blend with region 2
                        weight1 = compute_transition_weight(azim_normalized, 60, transition_width, True)
                        embed1 = azim_prompt(prompt_embed_dict, pose)
                        embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed1, embed2, weight1)
                        neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed1, neg_embed2, weight1)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight1: {weight1:.2f}")
                    elif azim_normalized > 300:
                        # Near 300° boundary, blend with region 3
                        weight1 = compute_transition_weight(azim_normalized, 300, transition_width, True)
                        embed1 = azim_prompt(prompt_embed_dict, pose)
                        embed2 = azim_prompt(tertiary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed1, embed2, weight1)
                        neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        neg_embed2 = azim_neg_prompt(tertiary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed1, neg_embed2, weight1)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight1: {weight1:.2f}")
                    else:
                        # Pure region 1
                        pos_embed = azim_prompt(prompt_embed_dict, pose)
                        neg_embed = azim_neg_prompt(negative_prompt_embed_dict, pose)
                elif azim_normalized > 60 and azim_normalized <= 180:
                    # Region 2: 60° → 180°
                    # Check if near boundaries
                    if azim_normalized <= 60 + transition_width:
                        # Near 60° boundary, blend with region 1
                        weight2 = compute_transition_weight(azim_normalized, 60, transition_width, False)
                        embed1 = azim_prompt(prompt_embed_dict, pose)
                        embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed1, embed2, 1.0 - weight2)
                        neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed1, neg_embed2, 1.0 - weight2)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight1: {1.0 - weight2:.2f}")
                    elif azim_normalized >= 180 - transition_width:
                        # Near 180° boundary, blend with region 3
                        weight2 = compute_transition_weight(azim_normalized, 180, transition_width, True)
                        embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                        embed3 = azim_prompt(tertiary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed2, embed3, weight2)
                        neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        neg_embed3 = azim_neg_prompt(tertiary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed2, neg_embed3, weight2)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight2: {weight2:.2f}")
                    else:
                        # Pure region 2
                        pos_embed = azim_prompt(secondary_prompt_embed_dict, pose)
                        neg_embed = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                elif azim_normalized > 180 and azim_normalized <= 300:
                    # Region 3: 180° → 300°
                    # Check if near boundaries
                    if azim_normalized <= 180 + transition_width:
                        # Near 180° boundary, blend with region 2
                        weight3 = compute_transition_weight(azim_normalized, 180, transition_width, False)
                        embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                        embed3 = azim_prompt(tertiary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed2, embed3, 1.0 - weight3)
                        neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        neg_embed3 = azim_neg_prompt(tertiary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed2, neg_embed3, 1.0 - weight3)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight2: {1.0 - weight3:.2f}")
                    elif azim_normalized >= 300 - transition_width:
                        # Near 300° boundary, blend with region 1
                        weight3 = compute_transition_weight(azim_normalized, 300, transition_width, False)
                        embed1 = azim_prompt(prompt_embed_dict, pose)
                        embed3 = azim_prompt(tertiary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed1, embed3, 1.0 - weight3)
                        neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        neg_embed3 = azim_neg_prompt(tertiary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed1, neg_embed3, 1.0 - weight3)
                        print(f"Blending prompts for view {i}, azimuth: {azim} (normalized: {azim_normalized}), weight1: {1.0 - weight3:.2f}")
                    else:
                        # Pure region 3
                        pos_embed = azim_prompt(tertiary_prompt_embed_dict, pose)
                        neg_embed = azim_neg_prompt(tertiary_negative_prompt_embed_dict, pose)
                        print(f"Using third prompt for view {i}, azimuth: {azim} (normalized: {azim_normalized})")
                
                pos_embeds_per_view.append(pos_embed)
                neg_embeds_per_view.append(neg_embed)
            else:
                # Original logic for other modes (quadrant, half) with smooth transition
                if self.config.azim_split_mode == 'quadrant':
                    # Prompt1: (270°, 360] U [0°, 90°]; prompt2: (90°, 270°]. Boundaries 90° / 270°.
                    boundaries = [90, 270]
                    
                    # Check if transition is disabled (hard switch mode)
                    if transition_width <= 0:
                        # Hard switch: no blending, use pure prompts based on region
                        if azim > 270 or azim <= 90:
                            # Region 1: use prompt1
                            pos_embed = azim_prompt(prompt_embed_dict, pose)
                            neg_embed = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        else:
                            # Region 2: use prompt2
                            pos_embed = azim_prompt(secondary_prompt_embed_dict, pose)
                            neg_embed = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        
                        pos_embeds_per_view.append(pos_embed)
                        neg_embeds_per_view.append(neg_embed)
                        continue
                    
                    # Smooth transition mode: use linear interpolation for symmetric weight distribution
                    # 0°: 1.0 prompt1, 45°: 0.8 prompt1, 90°: 0.5 prompt1, 135°: 0.2 prompt1, 180°: 0.0 prompt1
                    # 180°: 0.0 prompt1, 225°: 0.2 prompt1, 270°: 0.5 prompt1, 315°: 0.8 prompt1, 360°: 1.0 prompt1
                    
                    # Normalize azim to 0-360 range
                    azim_normalized = azim % 360
                    
                    # Piecewise linear interpolation based on key points
                    # 0°: 1.0, 45°: 0.8, 90°: 0.5, 135°: 0.2, 180°: 0.0
                    # 180°: 0.0, 225°: 0.2, 270°: 0.5, 315°: 0.8, 360°: 1.0
                    if azim_normalized <= 45:
                        # 0° to 45°: linear from 1.0 to 0.8
                        weight1 = 1.0 - (azim_normalized / 45.0) * 0.2
                    elif azim_normalized <= 90:
                        # 45° to 90°: linear from 0.8 to 0.5
                        weight1 = 0.8 - ((azim_normalized - 45.0) / 45.0) * 0.3
                    elif azim_normalized <= 135:
                        # 90° to 135°: linear from 0.5 to 0.2
                        weight1 = 0.5 - ((azim_normalized - 90.0) / 45.0) * 0.3
                    elif azim_normalized <= 180:
                        # 135° to 180°: linear from 0.2 to 0.0
                        weight1 = 0.2 - ((azim_normalized - 135.0) / 45.0) * 0.2
                    elif azim_normalized <= 225:
                        # 180° to 225°: linear from 0.0 to 0.2
                        weight1 = 0.0 + ((azim_normalized - 180.0) / 45.0) * 0.2
                    elif azim_normalized <= 270:
                        # 225° to 270°: linear from 0.2 to 0.5
                        weight1 = 0.2 + ((azim_normalized - 225.0) / 45.0) * 0.3
                    elif azim_normalized <= 315:
                        # 270° to 315°: linear from 0.5 to 0.8
                        weight1 = 0.5 + ((azim_normalized - 270.0) / 45.0) * 0.3
                    else:
                        # 315° to 360°: linear from 0.8 to 1.0
                        weight1 = 0.8 + ((azim_normalized - 315.0) / 45.0) * 0.2
                    
                    # Clamp to [0, 1]
                    weight1 = max(0.0, min(1.0, weight1))
                    
                    # Always blend (no pure regions except at exact boundaries)
                    if abs(weight1 - 1.0) < 1e-6:
                        # Pure prompt1 (0° or 360°)
                        pos_embed = azim_prompt(prompt_embed_dict, pose)
                        neg_embed = azim_neg_prompt(negative_prompt_embed_dict, pose)
                    elif abs(weight1 - 0.0) < 1e-6:
                        # Pure prompt2 (180°)
                        pos_embed = azim_prompt(secondary_prompt_embed_dict, pose)
                        neg_embed = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                    else:
                        # Blend prompts
                        embed1 = azim_prompt(prompt_embed_dict, pose)
                        embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                        pos_embed = blend_prompts(embed1, embed2, weight1)
                        neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                        neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                        neg_embed = blend_prompts(neg_embed1, neg_embed2, weight1)
                        print(f"Blending prompts for view {i}, azimuth: {azim}, weight1: {weight1:.2f}")
                    
                    pos_embeds_per_view.append(pos_embed)
                    neg_embeds_per_view.append(neg_embed)
                    
                elif self.config.azim_split_mode == 'half':
                    # Hemispheres: prompt1 [0°, 180°), prompt2 [180°, 360°); boundary 180°.
                    boundary = 180
                    
                    # Determine region and compute weights
                    if azim < 180:
                        # Region 1: 0° → 180°
                        if azim >= 180 - transition_width:
                            # Near 180° boundary, blend with region 2
                            weight1 = compute_transition_weight(azim, boundary, transition_width, True)
                            embed1 = azim_prompt(prompt_embed_dict, pose)
                            embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                            pos_embed = blend_prompts(embed1, embed2, weight1)
                            neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                            neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                            neg_embed = blend_prompts(neg_embed1, neg_embed2, weight1)
                            print(f"Blending prompts for view {i}, azimuth: {azim}, weight1: {weight1:.2f}")
                        else:
                            # Pure region 1
                            pos_embed = azim_prompt(prompt_embed_dict, pose)
                            neg_embed = azim_neg_prompt(negative_prompt_embed_dict, pose)
                    else:
                        # Region 2: 180° → 360°
                        if azim <= 180 + transition_width:
                            # Near 180° boundary, blend with region 1
                            weight2 = compute_transition_weight(azim, boundary, transition_width, False)
                            embed1 = azim_prompt(prompt_embed_dict, pose)
                            embed2 = azim_prompt(secondary_prompt_embed_dict, pose)
                            pos_embed = blend_prompts(embed1, embed2, 1.0 - weight2)
                            neg_embed1 = azim_neg_prompt(negative_prompt_embed_dict, pose)
                            neg_embed2 = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                            neg_embed = blend_prompts(neg_embed1, neg_embed2, 1.0 - weight2)
                            print(f"Blending prompts for view {i}, azimuth: {azim}, weight1: {1.0 - weight2:.2f}")
                        else:
                            # Pure region 2
                            pos_embed = azim_prompt(secondary_prompt_embed_dict, pose)
                            neg_embed = azim_neg_prompt(secondary_negative_prompt_embed_dict, pose)
                    
                    pos_embeds_per_view.append(pos_embed)
                    neg_embeds_per_view.append(neg_embed)
                else:
                    raise ValueError(f"Unknown azim_split_mode: {self.config.azim_split_mode}")

        positive_prompt_embeds = torch.stack(pos_embeds_per_view, axis=0)
        negative_prompt_embeds = torch.stack(neg_embeds_per_view, axis=0)
        
        func_params = {
            "guidance_scale": guidance_scale,
            "positive_prompt_embeds": positive_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "conditioning_images": conditioning_images,
            "group_metas": self.group_metas,
            "controlnet_keep": controlnet_keep,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "guess_mode": guess_mode,
            "ref_attention_end": ref_attention_end,
            "num_timesteps": num_timesteps,
            "cross_attention_kwargs": cross_attention_kwargs,
            "generator": generator,
            "predicted_variance": None,
            "exp": exp,
            "main_views": main_views,
            "cos_weighted": True,
            "background_colors": background_colors,
        }
        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                func_params["cur_index"] = i
                self.model.set_up_coefficients(t, self.config.sampling_method)
                
                if z_t_given:
                    if latent_tex.dim() == 4:
                        latent_tex = latent_tex.squeeze(0)
                    input_params = {
                        "zts": latent_tex,
                        "xts": latents 
                        }
                else:
                    input_params = {
                        "xts": latents
                        }
                
                if t > (1-multiview_diffusion_end)*num_timesteps:
                    res_dict = self.one_step_process(
                        input_params=input_params, 
                        timestep=t, 
                        alphas=alphas,
                        sigmas=sigmas,
                        case_name=case_name,
                        **func_params
                    )
                    
                    assert res_dict["x_t_1"] != None or res_dict["z_t_1"] != None

                    if res_dict["x_t_1"] != None:
                        latents = res_dict["x_t_1"]
                    else:
                        latents = self.forward_mapping(res_dict["z_t_1"], bg=None, add_noise=True, **func_params)
                    
                    if res_dict["z_t_1"] != None:
                        latent_tex = res_dict["z_t_1"]
                    else:
                        latent_tex = self.inverse_mapping(res_dict["x_t_1"], **func_params)

                    if res_dict["x0s"] != None:
                        pred_original_sample = res_dict["x0s"]
                    else:
                        pred_original_sample = [self.color_latents["fuchsia"].unsqueeze(0) for _ in range(latents.shape[0])]
                        pred_original_sample = torch.cat(pred_original_sample, dim=0)

                    intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))

                else:
                    noise_pred = self.compute_noise_preds(latents, t, **func_params)
                    step_results = self.model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]
                    latent_tex = self.inverse_mapping(latents, **func_params)

                    intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))

                if (1-t/num_timesteps) < control_guidance_start[0]:
                    controlnet_conditioning_scale = initial_controlnet_conditioning_scale
                elif (1-t/num_timesteps) > control_guidance_end[0]:
                    controlnet_conditioning_scale = controlnet_conditioning_end_scale
                else:
                    alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
                    controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

                if (1-t/num_timesteps) < shuffle_background_change:
                    background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
                elif (1-t/num_timesteps) < shuffle_background_end:
                    background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses))]
                else:
                    background_colors = background_colors

                if i % log_interval == log_interval-1 or t == 1:
                    if view_fast_preview:
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            images = latent_preview(latent_images.to(self.model._execution_device))
                            images = np.concatenate([img for img in images], axis=1)
                            decoded_results.append(images)
                        result_image = np.concatenate(decoded_results, axis=0)
                        numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.png")
                    else:
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            images = decode_latents(self.model.vae, latent_images.to(self.model._execution_device))

                            images = np.concatenate([img for img in images], axis=1)

                            decoded_results.append(images)
                        result_image = np.concatenate(decoded_results, axis=0)
                        numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.png")

                    if not t < (1-multiview_diffusion_end)*num_timesteps:
                        if tex_fast_preview:
                            if latent_tex.dim() == 4:
                                tex = latent_tex.clone().squeeze(0)
                            else:
                                tex = latent_tex.clone()
                            texture_color = latent_preview(tex[None, ...])
                            numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
                        else:
                            self.uvp_rgb.to(self.model._execution_device)
                            result_tex_rgb, result_tex_rgb_output = get_rgb_texture(
                                self.model.vae, 
                                self.uvp_rgb, 
                                pred_original_sample,
                                disable=self.config.disable_voronoi
                            )
                            
                            numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        latents = latents.type(self.model.vae.dtype)
        log_final_img = decode_latents(self.model.vae, latents.to(self.model._execution_device))
        for _i in range(log_final_img.shape[0]):
            numpy_to_pil(log_final_img[_i])[0].save(f"{self.result_dir}/generated_views_rgb_{_i:02d}.png")
        log_final_img = np.concatenate(log_final_img, axis=1) 
        numpy_to_pil(result_image)[0].save(f"{self.result_dir}/generated_views_rgb.png")

        result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.model.vae, self.uvp_rgb, latents, disable=self.config.disable_voronoi)
        self.uvp.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1,2,0))

        eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        self.uvp_rgb.set_texture_map(result_tex_rgb)

        textured_views = self.uvp_rgb.render_textured_views()
        for _i in range(len(textured_views)):
            torch_to_pil(textured_views[_i][:3]).save(f"{eval_dir}/final_averaged_images_{_i:02d}.png")

        textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
        textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
        v = numpy_to_pil(textured_views_rgb)[0]
        v.save(f"{self.result_dir}/textured_views_rgb.png")

        if self.config.save_gif:
            log_size = 256
            self.test_dir = f"{self.output_dir}/gif"
            os.makedirs(self.test_dir, exist_ok=True)
            
            test_azimuth = np.linspace(0, 360, 50).tolist()
            test_elevation = 0
            rgb_test_camera_poses = []
            for i in range(len(test_azimuth)):
                rgb_test_camera_poses.append((test_elevation, test_azimuth[i]))
            
            camera_centers = ((0,0,0),)
            self.uvp_rgb.set_cameras_and_render_settings(rgb_test_camera_poses, 
                                                        centers=camera_centers, 
                                                        camera_distance=2.0)
            
            _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
            self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False, disable=self.config.disable_voronoi)
            textured_views = self.uvp_rgb.render_textured_views()
            gif_imgs = []
            for _i in range(len(textured_views)):
                log_final_img = F.interpolate(textured_views[_i][:-1, ...].unsqueeze(0), 
                                            size=(log_size, log_size),
                                            mode="bilinear",
                                            antialias=True)
                pil_img = torch_to_pil(log_final_img)
                pil_img.save(f"{self.test_dir}/test_views_rgb_{_i}.png")
                gif_imgs.append(pil_img)

            save_path = os.path.join(self.test_dir, "dense_view.gif")
            img, *imgs = gif_imgs
            img.save(save_path, save_all=True, append_images=imgs, duration=75, loop=0)

            if self.config.sdedit:
                self.original_img_dir = f"{self.output_dir}/gif_original"
                os.makedirs(self.original_img_dir, exist_ok=True)
                self.uvp_rgb.set_texture_map(rgb_gt_tex_resized)

                test_azimuth = np.linspace(0, 360, 50).tolist()
                test_elevation = 0
                rgb_test_camera_poses = []
                for i in range(len(test_azimuth)):
                    rgb_test_camera_poses.append((test_elevation, test_azimuth[i]))
                
                camera_centers = ((0,0,0),)
                self.uvp_rgb.set_cameras_and_render_settings(
                    rgb_test_camera_poses, 
                    centers=camera_centers,
                    camera_distance=2.0
                )
                
                _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
                self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False, disable=self.config.disable_voronoi)
                textured_views = self.uvp_rgb.render_textured_views()
                gif_imgs = []
                for _i in range(len(textured_views)):
                    log_final_img = F.interpolate(textured_views[_i][:-1, ...].unsqueeze(0), 
                                                size=(log_size, log_size),
                                                mode="bilinear",
                                                antialias=True)
                    pil_img = torch_to_pil(log_final_img)
                    pil_img.save(f"{self.original_img_dir}/test_views_rgb_{_i}.png")
                    gif_imgs.append(pil_img)

                save_path = os.path.join(self.original_img_dir, "dense_view.gif")
                img, *imgs = gif_imgs
                img.save(save_path, save_all=True, append_images=imgs, duration=75, loop=0)
