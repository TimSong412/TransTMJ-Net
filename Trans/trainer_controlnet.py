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
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

class CtlNetTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        os.system(f"cp {__file__} {self.model_dir}/trainer_controlnet.py")

        self.diffuser = CtrlNetModel(args)
        # self.delighter_optim = torch.optim.Adam(self.delighter.parameters(), lr=args.delighter_lr)
        self.diffuser_optim = torch.optim.AdamW(
            self.diffuser.parameters(), lr=args.ns_lr)

        self.diffuser_scheduler = torch.optim.lr_scheduler.StepLR(
            self.diffuser_optim, args.lr_decay_steps, args.lr_decay)

        self.model_list = {
            'diffuser': self.diffuser,
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
        
        self.diffuser, self.diffuser_optim, self.diffuser_scheduler = self.accelerator.prepare(
            self.diffuser, self.diffuser_optim, self.diffuser_scheduler)

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

        

        condition_tensor = batch["CT"]
        ref_img = condition_tensor[:, 1:2, :, :].clone().repeat(1, 3, 1, 1)
        target_img = batch["MRI"].repeat(1, 3, 1, 1)
        condition_tensor = condition_tensor * 2.0 - 1.0
        with self.accelerator.autocast():
            denoise_loss_or_image = self.diffuser.forward(condition_tensor, ref_img, target_img, latent_train=(not self.args.RGB), RGB=self.args.RGB)
        
        self.diffuser_optim.zero_grad()
        if self.args.RGB:
            loss =  torch.nn.functional.l1_loss(denoise_loss_or_image, target_img)* 0.1
        else:
            loss = denoise_loss_or_image

        loss_dict = {
            'loss': loss,
            'denoise_loss': loss
        }

        self.accelerator.backward(loss)
        
        self.diffuser_optim.step()
        self.diffuser_scheduler.step()

        train_result = {
            'loss_dict': loss_dict,
            "ref_img": ref_img,
            "target_img": target_img,
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

                for bid in range(logdict['ref_img'].shape[0]):
                    self.writer.add_image(
                        f"img/{bid}_target", logdict['target_img'][bid], step)
                    self.writer.add_image(
                        f"img/{bid}_ref", logdict['ref_img'][bid], step)

                   
    @torch.no_grad()
    def validation(self, eval_datachunk, step):
        self.diffuser.eval()
      
    
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_psnr_gray = 0
        total_ssim_gray = 0
        total_lpips_gray = 0
        cnt = 0
        for ci, chunk in enumerate(eval_datachunk):
            dataloader = torch.utils.data.DataLoader(chunk, batch_size=self.args.val_batchsize, shuffle=False, num_workers=4, pin_memory=True)
            dataloader = self.accelerator.prepare(dataloader)
            torch.cuda.ipc_collect()
           
            for bi, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation at step {step}, chunk {ci}"):
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
                    image = self.diffuser.forward(condition_tensor, ref_img, target_img, num_inference_steps=20).clamp(0, 1)

                B, C, H, W = image.shape
                total_psnr += psnr(image, target_img).mean()*B
                total_ssim += ssim(image, target_img).mean()*B
                total_lpips += self.lpips(image, target_img).mean()*B
                image_gray = torchvision.transforms.functional.rgb_to_grayscale(image)
                target_img_gray = torchvision.transforms.functional.rgb_to_grayscale(target_img)
                total_psnr_gray += psnr(image_gray, target_img_gray).mean()*B
                total_ssim_gray += ssim(image_gray, target_img_gray).mean()*B
                total_lpips_gray += self.lpips(image_gray.repeat(1,3,1,1), target_img_gray.repeat(1,3,1,1)).mean()*B
                cnt += B

                seq_names = batch["seq"][0]
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                font_size = 30
                font = ImageFont.truetype(font_path, font_size)
                # put name text to the ref_img
                for bid in range(ref_img.shape[0]):
                    seq_name = Path(seq_names[bid]).parent.stem
                    # use PIL to put text on ref_img
                    ref_img_pil = torchvision.transforms.ToPILImage()(ref_img[bid].cpu())
                    draw = ImageDraw.Draw(ref_img_pil)
                    draw.text((5, 30), seq_name, fill=(255, 255, 255), font=font)
                    ref_img[bid] = torchvision.transforms.ToTensor()(ref_img_pil).to(ref_img.device)

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
        total_psnr_gray /= cnt
        total_ssim_gray /= cnt
        total_lpips_gray /= cnt


        if self.args.multiGPU:
            self.accelerator.log({"val_metric/psnr": total_psnr.item(), "val_metric/ssim": total_ssim.item(), "val_metric/lpips": total_lpips.item()}, step)
        else:
            self.writer.add_scalar("val_metric/psnr", total_psnr, step)
            self.writer.add_scalar("val_metric/ssim", total_ssim, step)
            self.writer.add_scalar("val_metric/lpips", total_lpips, step)
            self.writer.add_scalar("val_metric/psnr_gray", total_psnr_gray, step)
            self.writer.add_scalar("val_metric/ssim_gray", total_ssim_gray, step)
            self.writer.add_scalar("val_metric/lpips_gray", total_lpips_gray, step)
        


        self.diffuser.train()

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
        