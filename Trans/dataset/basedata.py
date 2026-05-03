import torch
import numpy as np
from pathlib import Path
import tqdm
import cv2


class BaseData(torch.utils.data.Dataset):
    def __init__(self, args, eval=False):
        self.args = args
        self.online = args.data_online
        self.datadir = Path(args.datadir[0])
        if eval and hasattr(args, 'val_datadir') and len(args.val_datadir) > 0:
            self.datadir = Path(args.val_datadir[0])

        self.eval = eval
        
        all_seq = sorted(self.datadir.glob('*'))
        all_seq = [seq for seq in all_seq if seq.is_dir()]

        if not (hasattr(args, 'val_datadir')):
            if eval:
                val_seq = []
                for sid, seq in enumerate(all_seq):
                    if str(sid) in args.val_dataseq:
                        val_seq.append(seq)
                all_seq = val_seq
            else:
                train_seq = []
                for sid, seq in enumerate(all_seq):
                    if str(sid) not in args.val_dataseq[2:]:
                        train_seq.append(seq)
                all_seq = train_seq
        print(f"Number of sequences: {len(all_seq)}")
        self.seq_valid_num = []

        self.data_dirs = []

        self.data_seqs = []
        if self.online:
            seq = None
            for seq in tqdm.tqdm(all_seq, desc="Loading sequences", postfix=str(seq)):
                CT_np = []
                MRI_np = []
                data_dir = seq / 'data'
                all_CT = sorted(data_dir.glob('ct*'))[6:-6]
                all_MRI = sorted(data_dir.glob('mri*'))[6:-6]
                for ct_file in all_CT:
                    ct = np.load(ct_file)
                    # crop the ct
                    # ct = ct[152:, :-152]
                    # resize to 512*512
                    ct = cv2.resize(ct, (512, 512), interpolation=cv2.INTER_LINEAR)
                    CT_np.append(ct)
                for mri_file in all_MRI:
                    mri = np.load(mri_file)
                    # mri = mri[152:, :-152]
                    mri = cv2.resize(mri, (512, 512), interpolation=cv2.INTER_LINEAR)
                    MRI_np.append(mri)
                self.data_seqs.append((CT_np, MRI_np))
                self.seq_valid_num.append(len(MRI_np)-2)
                self.data_dirs.append(data_dir)
        else:
            seq = None
            for seq in tqdm.tqdm(all_seq, desc="Loading sequences", postfix=str(seq)):
                data_dir = seq / 'data'
                CT_files = []
                MRI_files = []
                
                all_CT = sorted(data_dir.glob('ct*'))
                all_MRI = sorted(data_dir.glob('mri*'))
                for ct_file in all_CT:
                    
                    CT_files.append(ct_file)
                for mri_file in all_MRI:
                    # mri = np.load(mri_file)
                    MRI_files.append(mri_file)
                self.data_seqs.append((CT_files, MRI_files))
                self.seq_valid_num.append(len(MRI_files)-2)
        self.seq_valid_num = np.array(self.seq_valid_num)
        self.seq_valid_num_accum = np.cumsum(self.seq_valid_num)
        
        

        
    def normalize_CT(self, input):
        return np.clip(input, 0, 1023.0) / 1023.0

    
    def normalize_MRI(self, input):
        # return np.clip(input, 0, 2047) / 2047.0
        return np.clip(input, 0, 3500.0) / 3500.0


    def __len__(self):

        return np.sum(self.seq_valid_num)
    

    def __getitem__(self, idx):
        seq_idx = np.where(self.seq_valid_num_accum > idx)[0][0]
        seq_idx = int(seq_idx)
        CT, MRI = self.data_seqs[seq_idx]
        idx = int(idx - self.seq_valid_num_accum[seq_idx-1]) if seq_idx > 0 else int(idx)
        MRI = MRI[idx+1]
        CT0 = CT[idx]
        CT1 = CT[idx+1]
        CT2 = CT[idx+2]
        if not self.online:
            CT0 = np.load(CT0.__str__())
            CT1 = np.load(CT1.__str__())
            CT2 = np.load(CT2.__str__())
            MRI = np.load(MRI.__str__())
        CT = np.stack([CT0, CT1, CT2], axis=0)
        MRI = MRI[None, ...]

        CT = self.normalize_CT(CT)
        MRI = self.normalize_MRI(MRI)
        
        return {
            "CT": torch.from_numpy(CT).float(),
            "MRI": torch.from_numpy(MRI).float(),
            "seq": [str(self.data_dirs[seq_idx])]
        }



if __name__ == "__main__":
    import argparse
    import os
    import time
    import numpy as np
    import math
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_online", type=bool, default=True)
    parser.add_argument("--datadir", type=list, default=["data510/train"]) 

    args = parser.parse_args()

    args.val_datadir = ['data510/eval']

    res = {}

    dataset = BaseData(args, eval=True)
    print(f"Number of sequences: {len(dataset.data_seqs)}")
    print(f"Number of images: ", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    for bi, batch in enumerate(dataloader):
        # print(bi, batch["CT"].shape, batch["seq"])
        res[f"val_0_{bi}"] = batch['seq']
    # json.dump(res, open('val_data_raw.json', 'w'), indent=4)
