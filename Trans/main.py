import torch
import random
import numpy as np
import time
from pathlib import Path
import torch.utils
import tqdm
from accelerate import Accelerator
import yaml
import pdb
from torch.profiler import profile, record_function, ProfilerActivity
import gc


from config import get_args, setup_trainer
from dataset.create_dataset import create_dataset


def train(args):

    now = time.strftime("%y%m%d-%H%M", time.localtime())
    args.exp = f"{now}_{args.exp}"
    
    outdir = Path(args.outdir) / args.exp
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / 'configs.yaml', 'w') as f:
        config_str = yaml.dump(vars(args))
        f.write(config_str)

    args.outdir = Path(args.outdir)

    trainer = setup_trainer(args)

    if args.val_batchsize == 0:
        args.val_batchsize = args.batchsize

    train_data, eval_data = create_dataset(args)
    
    if isinstance(train_data, torch.utils.data.Dataset):
        train_data_chunks = [train_data]
        eval_data_chunks = [eval_data]
    else:
        train_data_chunks = train_data
        eval_data_chunks = eval_data
        print(f"Number of train chunks: {train_data.chunks()}")
        if args.eval:
            print(f"Number of eval chunks: {eval_data.chunks()}")


    torch.cuda.empty_cache()

    total_step = 0
    for epoch in range(1, args.epochs+1):
        with tqdm.tqdm(total=len(train_data)//args.batchsize, desc=f"Epoch {epoch}/{args.epochs}") as pbar:

            for train_dataset in train_data_chunks:

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)

                if hasattr(trainer, 'accelerator') and isinstance(trainer.accelerator, Accelerator):
                    trainer.accelerator.print(f"Accelerator using {trainer.accelerator.device.type} device")
                    train_loader = trainer.accelerator.prepare(train_loader)
                
                # with profile(activities=[
                #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                for i, batch in enumerate(train_loader):

                    res = trainer.train_one_step(batch)
                    
                
                    if total_step % 10 == 0:
                        trainer.log(res, total_step)
                    

                    pbar.set_postfix(loss=res["loss"])
                    pbar.update(1)

                    if args.eval:
                        # if total_step % args.log_steps == 0:
                        #     trainer.log(res, total_step, img=True)
                        if total_step % args.val_steps == 0:
                            del batch
                            if hasattr(trainer, 'accelerator') and isinstance(trainer.accelerator, Accelerator):
                                trainer.accelerator.free_memory()
                            torch.cuda.empty_cache()
                            logdict = trainer.validation(eval_data_chunks, total_step)
                    total_step += 1
                
                
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                del train_loader
                del train_dataset
                
                if hasattr(trainer, 'accelerator') and isinstance(trainer.accelerator, Accelerator):
                    trainer.accelerator.free_memory()

        if (epoch % args.save_steps == 0) and trainer.accelerator.is_main_process:
            ckptname = trainer.save_checkpoint(epoch, total_step)

       
                



def eval(args):

    now = time.strftime("%y%m%d-%H%M", time.localtime())
    args.exp = f"{now}_{args.exp}"
    
    outdir = Path(args.outdir) / args.exp
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / 'configs.yaml', 'w') as f:
        config_str = yaml.dump(vars(args))
        f.write(config_str)

    args.outdir = Path(args.outdir)

    trainer = setup_trainer(args)


    if args.val_batchsize == 0:
        args.val_batchsize = args.batchsize

    _, eval_data = create_dataset(args, eval_only=True)
    

    eval_data_chunks = [eval_data]


    torch.cuda.empty_cache()

    total_step = 0
   
    logdict = trainer.validation(eval_data_chunks, total_step)






if __name__ == '__main__':
    args = get_args()
    if args.eval_only:
        eval(args)
    else:
        train(args)