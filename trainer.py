import logging

import torch.nn as nn
import torch
import torchsummary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import numpy as np

class AETrainer():
    def __init__(self, args, dataset, testset, model, artifacts_dir):

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        self.data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        self.test_loader = DataLoader(testset, batch_size=16, shuffle=True)
        self.test_iter = iter(self.test_loader)

        self.device = torch.device(f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.is_available() else 'cpu')

        if len(dataset.dim) == 3:
            torchsummary.summary(model, tuple(dataset.dim), device='cpu', batch_size=args.batch_size)
        else:
            print(model)

        self.model = model.to(self.device)

        N = len(self.data_loader)
        total_steps = args.epochs * N

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=1e-3)
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps,
                                                                   eta_min=args.learning_rate/10)

        self.artifacts_dir = artifacts_dir
        self.writer = SummaryWriter(self.artifacts_dir)

        logging.info(f'Num batches per epoch: {N}; Total num batches: {args.epochs * N}; Device: `{self.device}`')

        self.logfile = open(self.artifacts_dir + '/logfile.txt', 'w')

        self.imagegetter = None
        self.args = args
        self.step = 0

    def log_model(self, log_scalars, log_embeddings,log_images=True):

        try:
            (title, x, y) = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_loader)
            (title, x, y) = next(self.test_iter)

        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        ŷ = self.model(x)
        
        if len(x.shape) == 5:
            t = x.shape[0]
            xNT = self.model.combine_NT(x)
            z =  self.model.encoder(xNT)
            z = self.model.split_NT(z, t)
        elif len(x.shape) == 4:
            z = self.model.encoder(x)

        self.model.train()

        loss = nn.functional.binary_cross_entropy(ŷ, y)
        
        if log_images:
            self.writer.add_video('X', x[:8], self.step)
            self.writer.add_video('Ŷ', ŷ[:8], self.step)
            self.writer.add_video('Y', y[:8], self.step)

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.artifacts_dir}/model_model_{self.step}.pth')

    def train(self):

        N = len(self.data_loader)
        total_steps = self.args.epochs * N
        step = 0

        pbar = tqdm(range(total_steps))
        pbar.set_description("Batch")
        for epoch in range(self.args.epochs):
            for i, (_, x, y) in enumerate(self.data_loader):

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                ŷ = self.model(x)

                loss = nn.functional.binary_cross_entropy(ŷ, y)

                loss.backward()
                self.optimizer.step()

                self.lr_sched.step()
                self.step = epoch * N + i

                log_scalars = self.step % self.args.log_step_scalars == 0
                log_embeddings = self.step % self.args.log_step_embeddings == 0
                if log_scalars or log_embeddings:
                    self.log_model(log_scalars, log_embeddings)

                pbar.update(self.step)

        self.log_model(True, True)
        self.save_model()
