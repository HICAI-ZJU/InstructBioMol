"""batch samplers that work with either random or sequential data samplers"""
import math
import os
import sys
import random

import torch
from torch.utils import data
import numpy as np


class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """

    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False,
                 gradient_accumulation_steps=None):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]



class DistributedWeightedAccumulateMultiDatasetBatchSampler(data.sampler.BatchSampler):
    """
    This is a modality-blended batch sampler which allows to sample a batch data from different dataset alternatively.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 dataset,
                 drop_last,
                 gradient_accumulation_steps,
                 selected_prob: list = None,
                 rank=-1,
                 world_size=2,
                 wrap_last=False):
        super(DistributedWeightedAccumulateMultiDatasetBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.wrap_last = wrap_last
        self.drop_last = drop_last
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets.datasets)
        self.largest_dataset_size = max([_cur_dataset.__len__() for _cur_dataset in dataset.datasets.datasets])

        if selected_prob is None:
            self.selected_prob = [1 / self.number_of_datasets for _ in range(self.number_of_datasets)]
        else:
            selected_prob = [float(p) for p in selected_prob]
            self.selected_prob = [p / sum(selected_prob) for p in selected_prob]

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets.datasets[dataset_idx]
            sampler = torch.utils.data.RandomSampler(cur_dataset)
            batch_sampler = DistributedBatchSampler(sampler, self.batch_size, self.drop_last, self.rank,
                                                    self.world_size, self.wrap_last)
            samplers_list.append(batch_sampler)
            cur_sampler_iterator = batch_sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.datasets.cumulative_sizes[:-1]
        step = self.batch_size * self.gradient_accumulation_steps
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        for _ in range(0, epoch_samples, step):
            # select `i`th dataset
            i = random.choices(range(self.number_of_datasets), self.selected_prob, k=1)[0]
            for _ in range(self.gradient_accumulation_steps):
                cur_batch_sampler = sampler_iterators[i]
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    # idx
                    cur_samples = [x + push_index_val[i] for x in cur_sample_org]
                    yield cur_samples
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    sampler_iterators[i] = samplers_list[i].__iter__()
                    cur_batch_sampler = sampler_iterators[i]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_samples = [x + push_index_val[i] for x in cur_sample_org]
                    yield cur_samples
