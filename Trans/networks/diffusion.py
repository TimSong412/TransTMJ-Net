from diffusers import DiffusionPipeline
import torch
import torchvision
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
import numpy as np
import tqdm
import open_clip
import matplotlib.pyplot as plt
import os
from torch.utils.checkpoint import checkpoint


def compute_diffmap(img0, img1, target_img0, target_img1, campose0, campose1, intrin0, intrin1, depth0, depth1, visualize=False):
    '''
    img0: torch.Tensor, B, 3, H, W
    img1: torch.Tensor, B, 3, H, W
    campose0: torch.Tensor, B, 4, 4
    campose1: torch.Tensor, B, 4, 4
    intrin0: torch.Tensor, B, 4
    intrin1: torch.Tensor, B, 4
    depth0: torch.Tensor, B, 1, H, W
    depth1: torch.Tensor, B, 1, H, W

    warp depth1 to depth0

    return: 
    diffmap: torch.Tensor, B, 1, H, W, including visible mask,
    '''
    B, _, H, W = img0.shape

    depth1[depth1 == depth1.max()] = 0
    depth_mask = depth1[:, 0] > 0
    full_mask = depth_mask.clone()

    uv1 = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    uv1 = torch.stack(uv1, dim=-1).to(depth1.device,
                                      dtype=depth1.dtype).unsqueeze(0).repeat(B, 1, 1, 1)
    uv1 -= intrin1[..., 2:].unsqueeze(-2).unsqueeze(-2)
    uv1 /= intrin1[..., :2].unsqueeze(-2).unsqueeze(-2)
    XYZ1 = torch.cat([uv1, depth1[:, 0].unsqueeze(-1)], dim=-1)
    XYZ1[..., 0] *= depth1[:, 0]
    XYZ1[..., 1] *= depth1[:, 0]
    XYZ1 = XYZ1.reshape(B, -1, 3)

    XYZ1_world = torch.matmul(XYZ1, campose1[:, :3, :3].transpose(
        1, 2)) + campose1[:, :3, 3].unsqueeze(1)
    XYZ1_0 = torch.matmul(
        (XYZ1_world - campose0[:, :3, 3].unsqueeze(1)), campose0[:, :3, :3])

    UV1_0 = XYZ1_0[..., :2] / XYZ1_0[..., 2:]
    UV1_0 *= intrin0[..., None, :2]
    UV1_0 += intrin0[..., None, 2:]
    in_range_mask = (UV1_0[..., 0] >= 0) & (UV1_0[..., 0] < W) & (
        UV1_0[..., 1] >= 0) & (UV1_0[..., 1] < H)
    in_range_mask = in_range_mask.reshape(B, H, W)

    # normalize to [-1, 1]
    UV1_0_normalized = UV1_0 / \
        torch.tensor([W, H]).to(UV1_0.device, dtype=UV1_0.dtype)
    UV1_0_normalized = UV1_0_normalized * 2.0 - 1.0

    target_depth = torch.nn.functional.grid_sample(
        depth0, UV1_0_normalized[:, None], mode='bilinear', align_corners=False).squeeze(1).squeeze(1)
    visible_mask = (target_depth - XYZ1_0[..., 2]) >= (0.0 - 2e-3)
    visible_mask = visible_mask.reshape(B, H, W)
    in_range_mask = in_range_mask & visible_mask
    full_mask = full_mask & in_range_mask

    query_color1 = torch.nn.functional.grid_sample(
        img0, UV1_0_normalized[:, None], mode='bilinear', align_corners=False).squeeze(2).reshape(B, 3, H, W)
    target_query_color1 = torch.nn.functional.grid_sample(
        target_img0, UV1_0_normalized[:, None], mode='bilinear', align_corners=False).squeeze(2).reshape(B, 3, H, W)

    if visualize:
        color_mark = torch.tensor(plt.cm.hsv(np.linspace(
            0, 1, W*H))[:, :3]).to(img0.device, dtype=img0.dtype)[full_mask[0].reshape(-1)]
        img0_vis = img0[0].clone()
        UV1_00 = UV1_0.round().long()[0][full_mask[0].reshape(-1)]
        img0_vis[:, UV1_00[:, 1], UV1_00[:, 0]] = color_mark.T
        img1_vis = img1[0].clone()
        img1_vis[:, full_mask[0]] = color_mark.T
        return img0_vis, img1_vis, query_color1, target_query_color1, full_mask

    return query_color1, target_query_color1, full_mask


def compute_diffuse_specular(target_light, ref_global_normal, ref_view_pose, ref_view_intrinsics):
    N, H, W = ref_global_normal.shape[0], ref_global_normal.shape[-2], ref_global_normal.shape[-1]
    lightmap = target_light.unsqueeze(-1).unsqueeze(-1)
    lightmap = lightmap.expand(-1, -1, H, W)

    fx, fy, cx, cy = ref_view_intrinsics[..., 0:1], ref_view_intrinsics[...,
                                                                        1:2], ref_view_intrinsics[..., 2:3], ref_view_intrinsics[..., 3:4]
    # compute view direction
    uv = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    uv = torch.stack(uv, dim=-1).to(ref_global_normal.device,
                                    dtype=ref_global_normal.dtype)
    uv = uv.unsqueeze(0).expand(ref_global_normal.shape[0], -1, -1, -1)
    uv = uv.reshape(N, -1, 2)
    xyz = torch.zeros_like(uv)
    xyz = torch.cat([xyz, torch.ones_like(xyz[..., 0:1])],
                    dim=-1).reshape(N, -1, 3)
    xyz[..., 0] = (uv[..., 0] - cx) / fx
    xyz[..., 1] = (uv[..., 1] - cy) / fy
    xyz = xyz / xyz.norm(dim=-1, keepdim=True)
    ref_view_dir = ref_view_pose[:, :3, :3] @ xyz.permute(0, 2, 1)  # in global
    ref_view_dir = ref_view_dir.reshape(N, 3, H, W) * -1.0

    # half vector
    half_vector = (lightmap + ref_view_dir) / (lightmap + ref_view_dir).norm(dim=1, keepdim=True) * \
        lightmap.norm(
            dim=1, keepdim=True)  # if light=(0, 0, 0), half_vector = (0, 0, 0)
    # compute specular
    specular = torch.sum(ref_global_normal * half_vector,
                         dim=1, keepdim=True)**10
    # compute diffuse
    diffuse = torch.sum(ref_global_normal * lightmap,
                        dim=1, keepdim=True).clamp(min=0)
    return diffuse, specular


def prepare_condition_tensor(args, ref_feat, target_lightmap_local, ref_img, diffuse, specular, shadow=None):
    '''
    ref_feat: torch.Tensor, B, F, H, W
    target_lightmap_local: torch.Tensor, B, 3, H, W
    ref_img: torch.Tensor, B, 3, H, W, [0, 1]
    diffuse: torch.Tensor, B, 1, H, W, [0, 1]
    specular: torch.Tensor, B, 1, H, W, [0, 1]
    shadow: torch.Tensor, B, 1, H, W, [0, 1]
    '''
    ref_img_normalized = (ref_img * 2.0 - 1.0)
    diffuse_normalized = (diffuse * 2.0 - 1.0)
    specular_normalized = (specular * 2.0 - 1.0)

    if args.use_feature:
        condition_tensor = torch.cat(
            [ref_feat, target_lightmap_local, ref_img_normalized, diffuse_normalized, specular_normalized], dim=1)
    else:
        condition_tensor = torch.cat(
            [target_lightmap_local, ref_img_normalized, diffuse_normalized, specular_normalized], dim=1)
    if args.use_shadow:
        shadow_normalized = (shadow * 2.0 - 1.0)
        condition_tensor = torch.cat(
            [condition_tensor, shadow_normalized], dim=1)
    return condition_tensor


class CtrlNetModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        os.system(f"cp {__file__} {self.args.model_dir}/diffusion.py")

        if hasattr(args, 'ckpt'):
            print(f"Loading model from {args.ckpt}")

            self.vae = AutoencoderKL.from_pretrained(
                args.ckpt, subfolder="vae")
            
            self.unet = UNet2DConditionModel.from_pretrained(
                args.ckpt, subfolder="unet")

            self.controlnet = ControlNetModel.from_pretrained(
                args.ckpt, subfolder="controlnet")
        else:
            self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="vae")

            self.unet = UNet2DConditionModel.from_pretrained(
                "stabilityai/stable-diffusion-2-1", subfolder="unet")
            
            self.controlnet = ControlNetModel.from_unet(
                self.unet, conditioning_channels=args.condition_dim)


        self.forw_scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="scheduler")

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H/14", pretrained='laion2b_s32b_b79k')
        # self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", args.device)
        self.clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.Normalize(mean=(
                0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        self.clip_model.eval()
        self.neg_prompt = 'lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        # self.neg_prompt_embed = self.clip_model.encode_text(clip.tokenize(self.neg_prompt)).unsqueeze(0).detach()

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.clip_model.requires_grad_(False)

        self.vae.eval()
        self.unet.eval()

        self.controlnet.train()
        print("Diffusion model loaded")

    def train_one_step(self, condition_tensor, base_img, target_img, RGB=False):
        '''
        condition_tensor: torch.Tensor, B, F, H, W
        base_img: torch.Tensor, B, 3, H, W, [0, 1]
        target_img: torch.Tensor, B, 3, H, W, [0, 1]
        '''
        # target_img = target_img.to(self.args.device, dtype=self.args.dtype)
        # condition_tensor.to(self.args.device, dtype=self.args.dtype)
        target_img_normalized = (target_img * 2.0 - 1.0)
        latent = self.vae.encode(target_img_normalized).latent_dist.sample()

        latent = latent * self.vae.config.scaling_factor
        noise = torch.randn_like(latent)
        B = latent.shape[0]
        timesteps = torch.randint(
            0, self.forw_scheduler.config.num_train_timesteps, (B,), device=latent.device)
        self.forw_scheduler.set_timesteps(1, device=latent.device)
        noisy_latents = self.forw_scheduler.add_noise(
            latent, noise, timesteps)  # .to(dtype=self.args.dtype)
        base_img_processed = self.clip_preprocess(base_img)
        with torch.no_grad():
            encoder_hidden_states = self.clip_model.encode_image(
                base_img_processed)[:, None]  # .to(dtype=self.args.dtype)
        # condition_tensor = condition_tensor.to(self.args.device, dtype=self.args.dtype)
        
        down_block_res_sample, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_tensor,
            return_dict=False
        )
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample for sample in down_block_res_sample
            ],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=RGB
        )[0]

        if RGB:
            target = target_img
            for i in range(len(latent)):
                latent[i:i+1] = self.forw_scheduler.step(
                    model_pred[i:i+1], timesteps[i], latent[i:i+1], return_dict=True).pred_original_sample
            # latent = self.forw_scheduler.step(
            #     model_pred, timesteps, latent, return_dict=True).pred_original_sample
            image = self.vae.decode(
                latent/self.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2.0 + 0.5).clamp(0, 1)
            return image
        
        else:

            if self.forw_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.forw_scheduler.config.prediction_type == "v_prediction":
                target = self.forw_scheduler.get_velocity(latent, noise, timesteps)

            loss = torch.nn.functional.mse_loss(model_pred, target)

        return loss

    def forward(self, condition_tensor, base_img, target_img=None, num_inference_steps=20, latent_train=False, RGB=False, contronet_only=True):
        if contronet_only:
            self.unet.eval()
            self.vae.eval()
            self.clip_model.eval()
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.clip_model.requires_grad_(False)
        if latent_train:
            return self.train_one_step(condition_tensor, base_img, target_img)
        elif RGB:
            return self.train_one_step(condition_tensor, base_img, target_img, RGB=True)
        else:
            return self.inference(condition_tensor, base_img, target_img, num_inference_steps)

    def inference(self, condition_tensor, base_img, target_img=None, num_inference_steps=20):
        '''
        condition_tensor: torch.Tensor, B, F, H, W
        base_img: torch.Tensor, B, 3, H, W, [0, 1]
        target_img: torch.Tensor, B, 3, H, W, [0, 1]

        return: torch.Tensor, B, 3, H, W, [0, 1]
        '''
        base_img_processed = self.clip_preprocess(base_img)
        with torch.no_grad():
            encoder_hidden_states = self.clip_model.encode_image(base_img_processed)[
                :, None]
        # latent = self.vae.encode(target_img).latent_dist.sample()
        B, C, H, W = base_img.shape
        latent = torch.randn((B, 4, 64, 64)).to(
            condition_tensor.device) * self.forw_scheduler.init_noise_sigma
        self.forw_scheduler.set_timesteps(
            num_inference_steps, device=latent.device)
        timesteps = self.forw_scheduler.timesteps
        for i, t in enumerate(timesteps):
            latent_model_input = self.forw_scheduler.scale_model_input(
                latent, t)
            if self.args.inference_use_checkpoint:
                down_block_res_sample, mid_block_res_sample = checkpoint(
                    function=self.controlnet, 
                    sample= latent_model_input,
                    timestep= t,
                    encoder_hidden_states= encoder_hidden_states,
                    controlnet_cond=condition_tensor,
                    return_dict= False,
                    use_reentrant=False
                    )
                noise_pred = checkpoint(
                    function=self.unet, 
                    sample=latent_model_input, 
                    timestep=t, 
                    encoder_hidden_states=encoder_hidden_states, 
                    down_block_additional_residuals=[
                        sample for sample in down_block_res_sample
                    ],
                    mid_block_additional_residual= mid_block_res_sample,
                    return_dict=False,

                    use_reentrant=False
                    )[0]
            else:
                down_block_res_sample, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_tensor,
                    return_dict=False
                )
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample for sample in down_block_res_sample
                    ],
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False
                )[0]

            latent = self.forw_scheduler.step(
                noise_pred, t, latent, return_dict=False)[0]
        image = self.vae.decode(
            latent/self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2.0 + 0.5).clamp(0, 1)
        return image


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    from argparse import Namespace
    args = Namespace(device="cuda", dtype=torch.float16)
    model = CtrlNetModel(args)
    opt = torch.optim.AdamW(model.controlnet.parameters(), lr=1e-4)
    baseimg = Image.open(
        "render/dataset/334e159972fd425d93f29d3c19c7f811/color/light000/view000.png")
    baseimg = torchvision.transforms.ToTensor()(
        baseimg).unsqueeze(0).to('cuda')[:, :3]
    target_img = Image.open(
        "render/dataset/334e159972fd425d93f29d3c19c7f811/color/light001/view000.png")
    target_img = torchvision.transforms.ToTensor()(
        target_img).unsqueeze(0).to('cuda')[:, :3]

    condition = np.load(
        "render/dataset/334e159972fd425d93f29d3c19c7f811/color/light000/view000_dinoL_MVC.npz")['feat']
    condition = torch.tensor(condition).unsqueeze(
        0).permute(0, 3, 1, 2).to('cuda').float()

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1
    )
    model = accelerator.prepare(model)

    model = accelerator.unwrap_model(model)

    # loss = model.train_one_step(condition, baseimg, target_img)

    # loss.backward()
    # opt.step()
    # opt.zero_grad()

    # print("Done")