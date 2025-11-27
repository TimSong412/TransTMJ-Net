import configargparse
# from trainer_single import SingleTrainer
from trainer_controlnet import CtlNetTrainer
from trainer_vae import VAETrainer
from trainer_vit import ViTTrainer
# from trainer_pair import PairTrainer
# from trainer_decodefeat import DecoderTrainer
# from trainer_decodepair import DecodePairTrainer
import torch
import os
import numpy as np
import random
torch.hub.set_dir("./torch_cache")
os.environ["HF_HOME"] = "./huggingface_cache"


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# yaml config file parser
def get_args(default_cfg='configs/default.yaml'):
    
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           default_config_files=[default_cfg])
    
    parser.add_argument('--config', is_config_file=True, help='config file path', default=default_cfg)
    
    parser.add_argument('--dbg', type=bool, default=True, help='debug mode default True')

    parser.add_argument('--exp', type=str, default='exp', help='experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--outdir', type=str, default='output', help='output directory')
    parser.add_argument('--eval_only', default=False, action='store_true', help='evaluation only')


    # training
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--trainer', type=str, default='single', help='trainer type')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--steps', type=int, default=30000, help='number of steps')
    parser.add_argument('--delighter_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ns_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.6, help='learning rate decay')
    parser.add_argument('--lr_decay_steps', type=int, default=500, help='learning rate decay steps')
    parser.add_argument('--multiGPU', default=False, action='store_true', help='multiGPU')
    parser.add_argument('--load_optim', default=False, action='store_true', help='load optimizer and scheduler')
    parser.add_argument('--inference_use_checkpoint', default=False, action='store_true', help='use checkpoint in diffusion')
    parser.add_argument('--RGB', default=False, action='store_true', help='use RGB diffuse supervision')
    


    # data
    parser.add_argument('--dataset', type=str, default='single', help='dataset type',  nargs='+')
    parser.add_argument('--datadir', type=str, default='', help='data dir, default for training', nargs='+')
    parser.add_argument('--val_datadir', type=str, default='', help='eval data dir', nargs='+')
    parser.add_argument('--view_first', default=False, action='store_true', help='iterate view first')
    parser.add_argument('--chunk_size', type=int, default=0, help='chunk size')
    parser.add_argument('--minibatch', type=int, default=1, help='number of images in minibatch')
    parser.add_argument('--train_datadir', type=str, default='', help='index file', nargs='+')
    parser.add_argument('--val_dataseq', type=str, default='', help='index file', nargs='+')
    parser.add_argument('--data_online', default=False, action='store_true', help='online data')

    # visualization
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize')
    parser.add_argument('--vis_length', type=int, default=10, help='visualization video batch size')

    # validation
    parser.add_argument('--val_batchsize', type=int, default=0, help='validation batch size')
    parser.add_argument('--eval', default=False, help='evaluation', action='store_true')
    parser.add_argument('--val_steps', type=int, default=300, help='evaluation gaps')
    parser.add_argument('--save_steps', type=int, default=1, help='save steps' )
    parser.add_argument('--log_steps', type=int, default=500, help='log steps')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path')
    parser.add_argument('--val_lightid', type=int, default=[3, 8], help='validation light id', nargs='+')
    parser.add_argument('--val_viewid', type=int, default=[1, 6], help='validation view id', nargs='+')

    # controlnet
    parser.add_argument('--condition_dim', type=int, default=3, help='condition dimension')
    parser.add_argument('--use_feature', default=False, action='store_true', help='use featuremap in condition tensor')
    parser.add_argument('--use_shadow', default=False, action='store_true', help='use shadowmap in condition tensor')


    args = parser.parse_args()
    return args

def setup_trainer(args):
    # if args.trainer == 'single':
    #     trainer = SingleTrainer(args)
    if args.trainer == 'controlnet':
        trainer = CtlNetTrainer(args)
    if args.trainer == 'vae':
        trainer = VAETrainer(args)
    if args.trainer == 'vit':
        trainer = ViTTrainer(args)
    # if args.trainer == 'pair':
    #     trainer = PairTrainer(args)
    # if args.trainer == 'decoder':
    #     trainer = DecoderTrainer(args)
    # if args.trainer == 'decodepair':
    #     trainer = DecodePairTrainer(args)
    return trainer


if __name__ == '__main__':
    args = get_args()
    print(args)
    print(args.dbg)
    print(args.batchsize)
  