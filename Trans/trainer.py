import torch
from pathlib import Path
import time
import yaml
from tensorboardX import SummaryWriter
from accelerate import Accelerator

class BaseTrainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if not isinstance(args.outdir, Path):
            args.outdir = Path(args.outdir)
        if not isinstance(args.exp, str):
            args.exp = str(args.exp)
        self.expdir = args.exp
        self.outdir = args.outdir / self.expdir
        self.ckpt_dir = self.outdir / 'ckpt'
        self.log_dir = self.outdir / 'log'
        self.model_dir = self.outdir / 'model'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir = self.model_dir

        if args.multiGPU:
            self.writer = None
        else:
            self.writer = SummaryWriter(str(self.log_dir))
            
        print("Experiment directory: ", self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.model_list = {}
        self.optim_list = {}
        self.scheduler_list = {}



    
    def save_checkpoint(self, epoch, step):
        checkpoint = {
            'epoch': epoch,
            'step': step,
        }
        if hasattr(self, 'accelerator') and isinstance(self.accelerator, Accelerator) and self.args.multiGPU:
            if not self.accelerator.is_main_process:
                return
        if hasattr(self, 'accelerator') and isinstance(self.accelerator, Accelerator):
            for key in self.model_list:
                checkpoint[key] = self.accelerator.unwrap_model(self.model_list[key]).state_dict()
        else:   
            for key in self.model_list:
                checkpoint[key] = self.model_list[key].state_dict()

        for key in self.optim_list:
            checkpoint[key] = self.optim_list[key].state_dict()
        for key in self.scheduler_list:
            checkpoint[key] = self.scheduler_list[key].state_dict()
        # torch.save(checkpoint, f'{self.ckpt_dir}/{epoch:05d}.pth')
        self.diffuser.vae.save_pretrained(f"{self.ckpt_dir}/epoch{epoch:05d}_step{step:06d}/vae")
        self.diffuser.unet.save_pretrained(f"{self.ckpt_dir}/epoch{epoch:05d}_step{step:06d}/unet")
        self.diffuser.controlnet.save_pretrained(f"{self.ckpt_dir}/epoch{epoch:05d}_step{step:06d}/controlnet")

        print(f"Saved checkpoint at epoch {epoch} and step {step} to {self.ckpt_dir}")
        return f'{self.ckpt_dir}/{epoch:05d}.pth'

    def load_checkpoint(self, ckptdir):
        print(f"Loading checkpoint from {ckptdir}")
        checkpoint = torch.load(ckptdir, weights_only=False)
        for key in self.model_list:
            self.model_list[key].load_state_dict(checkpoint[key])

        
        if self.args.load_optim:
            print("Loading optimizer and scheduler")
            for key in self.scheduler_list:
                self.scheduler_list[key].load_state_dict(checkpoint[key])
            for key in self.optim_list:
                self.optim_list[key].load_state_dict(checkpoint[key])
        else:
            print("Not loading optimizer and scheduler")
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} at step {checkpoint['step']}")
        print(f"ckpt has keys: {checkpoint.keys()}")
        return checkpoint['epoch']


    def validation(self):
        pass

    def visualize(self):
        pass

    def log(self, logdict:dict, step:int):
        for key, value in logdict.items():
            # scalar
            if isinstance(value, float):
                value = value.item()
                self.writer.add_scalar(key, value, step)
            # image
            elif isinstance(value, torch.Tensor):
                self.writer.add_image(key, value, step)
        


if __name__ == '__main__':
    from config import get_args
    args = get_args()
    trainer = BaseTrainer(args)
    print(trainer.expdir)