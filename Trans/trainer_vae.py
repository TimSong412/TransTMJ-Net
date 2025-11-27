import torch
from trainer import BaseTrainer
import torchvision
from utils.loss_utils import ssim, psnr
import tqdm
from networks.diffusion import CtrlNetModel, compute_diffmap, compute_diffuse_specular, prepare_condition_tensor
from accelerate import Accelerator
import imageio
import numpy as np
import os
import lpips
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, UNet2DConditionModel

class VAETrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        os.system(f"cp {__file__} {self.model_dir}/trainer_controlnet.py")

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="vae")
        # self.delighter_optim = torch.optim.Adam(self.delighter.parameters(), lr=args.delighter_lr)
        self.diffuser_optim = torch.optim.AdamW(
            self.vae.parameters(), lr=args.ns_lr)

        self.diffuser_scheduler = torch.optim.lr_scheduler.StepLR(
            self.diffuser_optim, args.lr_decay_steps, args.lr_decay)

        self.model_list = {
            'vae': self.vae,
        }
        self.optim_list = {
            'diffuser_optim': self.diffuser_optim,
        }
        self.scheduler_list = {
            'diffuser_scheduler': self.diffuser_scheduler,
        }
        self.accelerator = Accelerator(gradient_accumulation_steps=1,
                                       mixed_precision='fp16',
                                       log_with="tensorboard",
                                       project_dir=self.outdir)
        
        self.vae, self.diffuser_optim, self.diffuser_scheduler = self.accelerator.prepare(
            self.vae, self.diffuser_optim, self.diffuser_scheduler)

        self.lpips = lpips.LPIPS(net='vgg')
        
        self.lpips.eval()
        self.lpips.to(self.accelerator.device)

        if args.multiGPU:
            log_dict = {
                "loss/loss": 0,
                "loss/denoise_loss": 0,
                "val_metric/psnr": 0,
                "val_metric/ssim": 0,
                "val_metric/lpips": 0,
                "lr/ctrlnet_lr": 0
            }
            self.accelerator.init_trackers("./log", config=log_dict)
    


    def train_one_step(self, batch):
        '''
        all images are in [0, 1], should be normalized to [-1, 1] before feeding to the model
        '''
        self.vae.train()
        self.vae.encoder.train()
        self.vae.decoder.train()
        

        condition_tensor = batch["CT"]
        ref_img = condition_tensor[:, 1:2, :, :].clone().repeat(1, 3, 1, 1)
        target_img = batch["MRI"].repeat(1, 3, 1, 1)
        condition_tensor = condition_tensor * 2.0 - 1.0
        target_img_gt = target_img.clone().detach()
        target_img_gt.requires_grad = False
        # target_img.requires_grad = True

        self.diffuser_optim.zero_grad()

        with self.accelerator.autocast():
            
            latent = self.vae.encode(condition_tensor).latent_dist.sample()
            recons_img = self.vae.decode(latent).sample

            recons_img = (recons_img + 1.0) / 2.0

            loss = torch.nn.functional.mse_loss(recons_img, target_img_gt) * 0.1

        loss_dict = {
            'loss': loss,
            'denoise_loss': loss
        }

        self.accelerator.backward(loss)
        
        self.diffuser_optim.step()
        self.diffuser_scheduler.step()

        train_result = {
            'loss_dict': loss_dict,
            "input_img": target_img,
            "recon_img": recons_img,
            'loss': loss.item(),
        }

        return train_result


    def log(self, logdict, step, img=False):

        if self.args.multiGPU:
            for key in logdict['loss_dict']:
                self.accelerator.log({f"loss/{key}": logdict['loss_dict'][key].item()}, step)
            self.accelerator.log({f"lr/ctrlnet_lr": self.diffuser_scheduler.get_last_lr()[0]}, step)
        else:
            for key in logdict['loss_dict']:
                self.writer.add_scalar(
                    "loss/"+key, logdict['loss_dict'][key].item(), step)
            if img:

                for bid in range(logdict['input_img'].shape[0]):
                    self.writer.add_image(
                        f"img/{bid}_recon", logdict['recon_img'][bid], step)
                    self.writer.add_image(
                        f"img/{bid}_input", logdict['input_img'][bid], step)

                   
    @torch.no_grad()
    def validation(self, eval_datachunk, step):
        self.vae.eval()
      
    
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        cnt = 0
        for ci, chunk in enumerate(eval_datachunk):
            dataloader = torch.utils.data.DataLoader(chunk, batch_size=self.args.val_batchsize, shuffle=False, num_workers=4, pin_memory=True)
            dataloader = self.accelerator.prepare(dataloader)
            torch.cuda.ipc_collect()
           
            for bi, batch in enumerate(dataloader):
                self.accelerator.free_memory()
                torch.cuda.empty_cache()
                # val_targetlight = batch['val_targetlight']
                # val_srclight = batch['val_srclight']
                # val_view = batch['val_view']


                condition_tensor = batch["CT"]
                ref_img = condition_tensor[:, 1:2, :, :].clone().repeat(1, 3, 1, 1)
                target_img = batch["MRI"].repeat(1, 3, 1, 1)

                condition_tensor = condition_tensor * 2.0 - 1.0


                with self.accelerator.autocast():
                    latent = self.vae.encode(condition_tensor).latent_dist.sample()
                    image = self.vae.decode(latent).sample
                    image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

                B, C, H, W = image.shape
                total_psnr += psnr(image, target_img).mean()*B
                total_ssim += ssim(image.float(), target_img).mean()*B
                total_lpips += self.lpips(image, target_img).mean()*B
                cnt += B

                if not self.args.multiGPU:
                    self.vis_dir = self.outdir / 'vis' / f"step_{step}"
                    self.vis_dir.mkdir(parents=True, exist_ok=True)
                    # self.writer.add_image(f"val/{ci}_{bi}_ref_image", ref_img[0], step)
                    # self.writer.add_image(f"val/{ci}_{bi}_denoise_image_{ci}_{bi}", image[0], step)
                    # self.writer.add_image(f"val/{ci}_{bi}_target_{ci}_{bi}", target_img[0], step)
                    full_image = torch.cat([ref_img, image, target_img], dim=2)
                    BW_image = torchvision.transforms.functional.rgb_to_grayscale(full_image)
                    torchvision.utils.save_image(
                        BW_image, self.vis_dir / f"val_{ci}_{bi}.png")
                del batch
                
                
            del dataloader
            del chunk
            self.accelerator.free_memory()
            torch.cuda.empty_cache()
            

        total_psnr /= cnt
        total_ssim /= cnt
        total_lpips /= cnt

        if self.args.multiGPU:
            self.accelerator.log({"val_metric/psnr": total_psnr.item(), "val_metric/ssim": total_ssim.item(), "val_metric/lpips": total_lpips.item()}, step)
        else:
            self.writer.add_scalar("val_metric/psnr", total_psnr, step)
            self.writer.add_scalar("val_metric/ssim", total_ssim, step)
            self.writer.add_scalar("val_metric/lpips", total_lpips, step)
        


        self.vae.train()

        torch.cuda.empty_cache()
        return {
            "metric": {
                "psnr": total_psnr.item() if isinstance(total_psnr, torch.Tensor) else total_psnr,
                "ssim": total_ssim.item() if isinstance(total_ssim, torch.Tensor) else total_ssim,
                "lpips": total_lpips.item() if isinstance(total_lpips, torch.Tensor) else total_lpips
            }
        }




    @torch.no_grad()    
    def visualize(self, vis_datachunk, render_ref_lightid=[1]):
        self.diffuser.eval()
       
        self.vis_dir = self.outdir / 'vis'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        