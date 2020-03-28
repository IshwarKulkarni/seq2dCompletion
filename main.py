import trainer
import datasets
import model
from datetime import datetime
import argparse
import torch

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--float_type", default='full', choices=['half', 'full'])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--layer_sizes", type=list, default=[128, 48, 32, 32, 16, 8])
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--log_step_scalars", type=int, default=50)
    parser.add_argument("--log_step_embeddings", type=int, default=250)
    parser.add_argument("--log_tsne_embedding", type=bool, default=False)

    return parser.parse_args(args="")

if __name__ == "__main__":
    args = get_default_args()
    dataset = datasets.SpiralDataset(3072, float_type=args.float_type)
    test_dataset = datasets.SpiralDataset(512,float_type=args.float_type)
    model = model.AutoEncoder(in_shape=dataset.dim,
                              mlp_layer_sizes=args.layer_sizes,
                              latent_size=args.latent_size,
                              is_variational=True, conditional_size=0)
    dt = datetime.now().strftime("%B-%d-%H:%M:%S")
    trnr = trainer.AETrainer(args, dataset, test_dataset, model,
                            args.float_type, f'artifacts/{dt}')

    try:
        trnr.train()
    except KeyboardInterrupt:
        if trnr:
            print("Keyboard Interrupt, saving model and quitting\n")
            trnr.save_model()


