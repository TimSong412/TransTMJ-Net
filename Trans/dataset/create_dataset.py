import torch
from .basedata import BaseData
import bisect
import warnings
from typing import (
    Iterable,
    List,
    Optional,
    TypeVar,
)
from torch.utils.data import Dataset, Sampler, IterableDataset


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


dataset_dict = {
    "base": BaseData,
}

def split_train_eval(dataset_type, args, eval_only=False):
    if args.eval and not eval_only:
        if args.chunk_size > 0:
            train_dataset = ChunkDataManager(args, eval=False)
            eval_dataset = ChunkDataManager(args, eval=True)
        else:
            train_dataset = dataset_dict[dataset_type](args, eval=False)
            eval_dataset = dataset_dict[dataset_type](args, eval=True)
        return train_dataset, eval_dataset
    elif eval_only:
        if args.chunk_size > 0:
            return None, ChunkDataManager(args, eval=True)
        else:
            return None, dataset_dict[dataset_type](args, eval=True)
    else:
        if args.chunk_size > 0:
            return ChunkDataManager(args, eval=False), None
        else:
            return dataset_dict[dataset_type](args, eval=False), None

def create_dataset(args, eval_only=False):
    for dataset_type in args.dataset:
        if dataset_type not in dataset_dict:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        trainset, evalset = split_train_eval(dataset_type, args, eval_only=eval_only)
        return trainset, evalset



class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def increase_max_interval_by(self, increment):
        for dataset in self.datasets:
            curr_max_interval = dataset.max_interval.value
            dataset.max_interval.value = min(curr_max_interval + increment, dataset.num_imgs - 1)

    def set_max_interval(self, max_interval):
        for dataset in self.datasets:
            dataset.max_interval.value = min(max_interval, dataset.num_imgs - 1)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
    

# class ChunkDataManager():
#     '''
#     concurrently init data batches
#     '''
#     def __init__(self, args, eval=False):
#         self.args = args
#         self.eval = eval
#         self.index_dict = read_index(args, eval=eval, visualize=args.visualize)
#         self.chunk_size = args.chunk_size
#         self.num_data = self.index_dict["num_data"]
#         self.num_views = sum(self.index_dict["data_num_view"])
#         # assert all views have same num light
#         assert len(set(self.index_dict["data_num_light"])) == 1
#         self.num_lights = self.index_dict["data_num_light"][0]
#         self.data_len = self.index_dict["data_len"]

#         self.chunk_viewids = split_chunks(args, self.index_dict, chunk_size=self.chunk_size, shuffle=((not eval) and (not args.visualize)))
#         self.chunk_nums = len(self.chunk_viewids)
#         self.current_chunk = 0 # start from 1 to chunk_nums
#         self.next_chunk_container = []
#         self.next_chunk_thread = None
#         self.reshuffle = False
    
#     def __len__(self):
#         return self.data_len

#     def __iter__(self):
#         self.reshuffle = False
#         return self
    
#     def __next__(self): 
#         if self.reshuffle:
#             raise StopIteration
#         dataset = self.next_chunk()
#         return dataset

#     def chunks(self):
#         return self.chunk_nums

#     def __del__(self):
#         if self.next_chunk_thread is not None:
#             self.next_chunk_thread.join()
    
#     def next_chunk(self):
#         if self.next_chunk_thread is None:
#             self.reshuffle = self.prepare_next_chunk()
#         dataset = self.join_next_chunk()
#         self.next_chunk_container.clear()
#         self.reshuffle = self.prepare_next_chunk()
#         gc.collect()
#         return dataset
        

#     def prepare_next_chunk(self):
#         '''
#         concurrently load a batch of data
#         '''
#         reshuffle = False
#         self.current_chunk += 1
#         if self.current_chunk > self.chunk_nums:
#             self.chunk_viewids = split_chunks(self.args, self.index_dict, chunk_size=self.chunk_size, shuffle=(not self.eval))
#             reshuffle = True
#             self.current_chunk = 1
#             self.chunk_nums = len(self.chunk_viewids)
#         batch_viewid = self.chunk_viewids[self.current_chunk-1]
#         self.next_chunk_thread = threading.Thread(target=self.load_batch, args=(batch_viewid, self.eval, self.next_chunk_container))
#         self.next_chunk_thread.start()
#         return reshuffle

#     def join_next_chunk(self):
#         '''
#         join the next batch thread
#         '''
#         self.next_chunk_thread.join()
#         self.next_chunk_thread = None
#         return self.next_chunk_container[0]


#     def load_batch(self, batch_viewid, eval, return_container:list):
#         '''
#         load a batch of data
#         '''
#         return_container.append(dataset_dict[self.args.dataset[0]](self.args, self.index_dict, batch_viewid, eval))

