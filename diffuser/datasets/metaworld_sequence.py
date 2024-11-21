import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import metaworld
import random

from .sequence import SequenceDataset
from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
from .preprocessing import get_preprocess_fn

class MetaworldSequenceDataset(SequenceDataset):
    def __init__(self, env, data_path, horizon=64,
                 normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
                 max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        mt = metaworld.ML1(env)
        env = mt.train_classes[env]()
        task = random.choice(mt.test_tasks)
        env.set_task(task)
        self.env = env
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.seed = seed

        # Load data from the pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Build the ReplayBuffer
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i in range(len(data['observations'])):
            episode = {
                'observations': data['observations'][i],
                'actions': data['actions'][i],
                'rewards': data['rewards'][i],
                'terminals': data['terminals'][i],
                'timeouts': data['timeouts'][i],
            }
            fields.add_path(episode)
        fields.finalize()

        # Check if the provided normalizer is already an instance of DatasetNormalizer
        if isinstance(normalizer, DatasetNormalizer):
            # Use the provided normalizer directly (e.g., from the training dataset)
            self.normalizer = normalizer
        else:
            # Otherwise, create a new normalizer based on the type string
            self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])

        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)