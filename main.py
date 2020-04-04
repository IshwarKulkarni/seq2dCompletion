import argparse
import json
import sys
from datetime import datetime

import numpy as np
import torch

import datasets
import models
import trainer

def get_default_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    #Trainer args
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

    #Model args
    parser.add_argument("--num_train_samples", type=int, default=3072)
    parser.add_argument("--num_test_samples", type=int, default=512)
    parser.add_argument("--layer_sizes", type=list, default=[128, 256, 128, 96, 64, 32])
    parser.add_argument("--latent_size", type=int, default=8)

    #Logging args
    parser.add_argument("--log_step_scalars", type=int, default=250)
    parser.add_argument("--log_step_embeddings", type=int, default=1000)
    parser.add_argument("--log_tsne_embedding", type=bool, default=False)

    if len(sys.argv) > 1:
        return parser.parse_args()
    return parser.parse_args(args="")


if __name__ == "__main__":
    args = get_default_args()
    print(args)
    experiment = 'TCHW'

    if experiment == 'CHW':
        dataset = datasets.SpiralDataset(args.num_train_samples)
        test_dataset = datasets.SpiralDataset(args.num_test_samples)
        model = models.AutoEncoder(in_shape=dataset.dim,
                                   conv_layer_sizes=args.layer_sizes, 
                                   enc_out_size=7, decoder_in_size=7)

    if experiment == 'TCHW':
        dataset = datasets.SpiralDatasetTCHW(args.num_train_samples)
        test_dataset = datasets.SpiralDatasetTCHW(args.num_test_samples)
        model = models.Recurrent(in_shape=dataset.dim,
                                 conv_layer_sizes=args.layer_sizes,
                                 batch_size=args.batch_size,
                                 out_t=dataset.t_out,
                                 rnn_z_size=7, seq_len=dataset.t_in)

    dt = datetime.now().strftime("%B-%d-%H:%M:%S")
    trnr = trainer.AETrainer(args, dataset, test_dataset, model,
                             f'artifacts/{dt}')

    try:
        trnr.train()
    except KeyboardInterrupt:
        if trnr:
            print("Keyboard Interrupt, saving model and quitting\n")
            trnr.save_model()
