import os
import wandb
import torch
import argparse

from math import inf
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import create_dataset
from pytorch_utils import heights_to_grid_masks, heights_to_latent_masks, \
    remove_temporal_padding
from vae import AutoEncoder, reconstruction_loss, kl_loss


def load_dataset(sequence_stride, batch_size, num_simulations=inf):
    """
    Load a dataset arranged for the (variational) auto-encoder.

    args:
    -----
        sequence_stride: int
            The sequence stride with which sequences should be under-sampled, along
            the temporal axis.
        batch_size: int
            The number of grids per batch.
        num_simulations: int (inf)
            The number of simulation sequences to load into the dataset, from the
            entire set of sequences. inf loads all available simulations.

    returns:
    --------
        train_set: DataLoader instance
            The training DataLoader instance for training a VAE. Each element of
            a batch is a tuple of (grids, heights) pairs, where grids is a
            tensor of size [1, H_max, W] representing the grid of temperatures
            and heights is a tensor of size [1] corresponding to the actual
            height of the grid.
        valid_set: DataLoader instance
            The validation DataLoader instance for training a VAE. Each element
            of a batch is a tuple of (grids, heights) pairs, where grids is a
            tensor of size [1, H_max, W] representing the grid of temperatures
            and heights is a tensor of size [1] corresponding to the actual
            height of the grid.
        test_set: DataLoader instance
            The testing DataLoader instance for training a VAE. Each element
            of a batch is a tuple of (grids, heights) pairs, where grids is a
            tensor of size [1, H_max, W] representing the grid of temperatures
            and heights is a tensor of size [1] corresponding to the actual
            height of the grid.
        mean_grids: tensor of size [1, H_max, W]
            The tensor of mean temperatures, for the training set.
        std_grids: tensor of size [1, H_max, W]
            The tensor of std temperatures, for the training set.
    """

    # Create train, val and test datasets with appropriate stride and window size.
    # Since we are training a VAE, we are only interested in grid transitions.
    dataset = create_dataset(sequence_stride=sequence_stride,
                             window_size=0,
                             use_transitions=True,
                             num_simulations=num_simulations)

    # Add a channel dimension
    train_grids = torch.tensor(
        dataset['train_seq_grids']).float().unsqueeze(1)
    valid_grids = torch.tensor(
        dataset['valid_seq_grids']).float().unsqueeze(2)
    test_grids = torch.tensor(
        dataset['test_seq_grids']).float().unsqueeze(2)

    train_heights = torch.tensor(
        dataset['train_seq_heights']).float()
    valid_heights = torch.tensor(
        dataset['valid_seq_heights']).float()
    test_heights = torch.tensor(dataset['test_seq_heights']).float()

    # Compute mean and std of grids of temperatures
    std_grids, mean_grids = torch.std_mean(train_grids, dim=0)

    _, _, max_height, width = train_grids.shape

    # Remove unnecessary temporal padding from valid grids (useless transitions)
    new_valid_grids = []
    new_valid_heights = []
    for seq_grids, seq_heights in zip(valid_grids, valid_heights):
        new_seq_grids, new_seq_heights = remove_temporal_padding(
            seq_grids, seq_heights)

        new_valid_grids.append(new_seq_grids)
        new_valid_heights.append(new_seq_heights)

    # Merge batch and temporal dimensions
    valid_grids = torch.cat(new_valid_grids, dim=0)
    valid_heights = torch.cat(new_valid_heights, dim=0)

    # Remove unnecessary temporal padding from test grids (useless transitions)
    new_test_grids = []
    new_test_heights = []
    for seq_grids, seq_heights in zip(test_grids, test_heights):
        new_seq_grids, new_seq_heights = remove_temporal_padding(
            seq_grids, seq_heights)

        new_test_grids.append(new_seq_grids)
        new_test_heights.append(new_seq_heights)

    # Merge batch and temporal dimensions
    test_grids = torch.cat(new_test_grids, dim=0)
    test_heights = torch.cat(new_test_heights, dim=0)

    # Prepare initial grid temperatures and initial height for training
    initial_grid = torch.full((1, 1, max_height, width), 20.)
    initial_height = torch.full((1,), max_height)
    train_grids = torch.cat([initial_grid, train_grids], dim=0)
    train_heights = torch.cat(
        [initial_height, train_heights], dim=0)

    # Create the DataLoader instances
    train_set = DataLoader(TensorDataset(train_grids, train_heights),
                           batch_size=batch_size, shuffle=True)

    valid_set = DataLoader(TensorDataset(valid_grids, valid_heights),
                           batch_size=batch_size, shuffle=True)

    test_set = DataLoader(TensorDataset(test_grids, test_heights),
                          batch_size=batch_size, shuffle=True)

    return (train_set, valid_set, test_set,
            mean_grids, std_grids)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--num_simulations', type=int, default=inf,
                        help='Number of simulations to load from the '
                             'dataset (for debug purposes), default is all '
                             'simulations.')
    parser.add_argument('--sequence_stride', type=int, default=10,
                        help='Sequence stride for the dataset, default is 10.')
    parser.add_argument('--normalize', action='store_true',
                        help='Whether to normalize the inputs or not.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model, default is 10.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size, default is 32.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate, default is 0.001.')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='Weight of the KL divergence loss, default is 0.1.')
    parser.add_argument('--lr_scheduling', action='store_true',
                        help='Whether to use learning rate scheduling or not.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Whether to use GPU or not.')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping value, default is None.')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension, default is 64.')
    parser.add_argument('--num_convs', type=int, default=3,
                        help='Number of convolutional layers, default is 3.')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Number of hidden channels of the first '
                        'convolutional layer. The i-th convolutional layer will '
                        'have "hidden_channels * (2 ** (i - 1))" channels. '
                        'Default is 16.')
    parser.add_argument('--num_fcs', type=int, default=2,
                        help='Number of fully connected layers, default is 2.')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden units of the fully connected '
                             'layers, default is 128.')
    parser.add_argument('--variational', action='store_true',
                        help='Whether to use a variational autoencoder or not.')
    parser.add_argument('--model_dir', type=str, default='models/vae',
                        help='Directory where the model will be saved, '
                        'default is "models/vae".')
    args = parser.parse_args()

    return args


def main(args):
    # Initialize wandb project
    wandb.init(project='vae', config=args)
    wandb.define_metric('epoch')
    wandb.define_metric('*', step_metric='epoch')

    # Detect device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('Using device:', device, flush=True)

    if wandb.run.name is None:
        filename = 'model.pt'

    else:
        filename = wandb.run.name + '.pt'

    model_file = os.path.join(args.model_dir, filename)

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Load train, valid and test dataset
    (train_set, valid_set, test_set,
     mean_grids, std_grids) = load_dataset(sequence_stride=args.sequence_stride,
                                           batch_size=args.batch_size,
                                           num_simulations=args.num_simulations)

    batch_grids, _ = next(iter(train_set))
    _, _, max_height, width = batch_grids.shape

    if not args.normalize:
        mean_grids, std_grids = None, None

    # Initialize model
    model = AutoEncoder(
        input_channels=1, input_shape=(max_height, width),
        latent_dim=args.latent_dim,
        num_convs=args.num_convs, hidden_channels=args.hidden_channels,
        num_fcs=args.num_fcs, hidden_size=args.hidden_size,
        mean_inputs=mean_grids, std_inputs=std_grids,
        variational=args.variational,
    )

    # Send model to device
    model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    if args.lr_scheduling:
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # Training loop
    best_valid_loss = inf
    for e in range(args.epochs):
        model.train()

        epoch_recons_loss = 0.
        num_temperatures = 0

        if args.variational:
            epoch_kl_loss = 0.
            num_latents = 0

        for batch_grids, batch_heights in train_set:
            batch_grids = batch_grids.to(device)
            batch_heights = batch_heights.to(device)

            batch_preds = model(batch_grids)

            # Unpack batch_preds if variational mode
            if args.variational:
                batch_preds, batch_means, batch_log_vars = batch_preds

            # Compute target masks
            batch_grid_masks = heights_to_grid_masks(
                batch_preds.shape, batch_heights)

            # Compute the masked reconstruction loss
            recons_loss = reconstruction_loss(batch_preds, batch_grids)
            recons_loss = recons_loss * batch_grid_masks
            recons_loss = recons_loss.sum()

            loss = recons_loss / batch_grid_masks.sum()

            # Compute latent masks (masks are for useless target grids)
            if args.variational:
                batch_latent_masks = heights_to_latent_masks(
                    batch_means.shape, batch_heights)

                kl = kl_loss(batch_means, batch_log_vars)
                kl = kl * batch_latent_masks
                kl = kl.sum()

                loss = loss + args.kl_weight * kl / batch_latent_masks.sum()

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_recons_loss += recons_loss.item()
            num_temperatures += batch_grid_masks.sum().item()

            if args.variational:
                epoch_kl_loss += kl.item()
                num_latents += batch_latent_masks.sum().item()

        if args.lr_scheduling:
            scheduler.step()

        # Normalize total epoch reconstruction and KL losses
        epoch_recons_loss /= num_temperatures
        epoch_loss = epoch_recons_loss

        if args.variational:
            epoch_kl_loss /= num_latents
            epoch_loss += args.kl_weight * epoch_kl_loss

        # Validation phase
        model.eval()

        valid_recons_loss = 0.
        num_temperatures = 0

        if args.variational:
            valid_kl_loss = 0.
            num_latents = 0

        with torch.no_grad():
            for batch_grids, batch_heights in valid_set:
                batch_grids = batch_grids.to(device)
                batch_heights = batch_heights.to(device)

                batch_preds = model(batch_grids)

                # Unpack batch_preds if variational mode
                if args.variational:
                    batch_preds, batch_means, batch_log_vars = batch_preds

                # Compute target masks
                batch_grid_masks = heights_to_grid_masks(
                    batch_preds.shape, batch_heights)

                # Compute the masked reconstruction loss
                recons_loss = reconstruction_loss(batch_preds, batch_grids)
                recons_loss = recons_loss * batch_grid_masks
                recons_loss = recons_loss.sum()

                # Compute latent masks (masks are for useless target grids)
                if args.variational:
                    batch_latent_masks = heights_to_latent_masks(
                        batch_means.shape, batch_heights)

                    kl = kl_loss(batch_means, batch_log_vars)
                    kl = kl * batch_latent_masks
                    kl = kl.sum()

                valid_recons_loss += recons_loss.item()
                num_temperatures += batch_grid_masks.sum().item()

                if args.variational:
                    valid_kl_loss += kl.item()
                    num_latents += batch_latent_masks.sum().item()

        # Normalize total epoch reconstruction and KL losses
        valid_recons_loss /= num_temperatures
        valid_loss = valid_recons_loss

        if args.variational:
            valid_kl_loss /= num_latents
            valid_loss += args.kl_weight * valid_kl_loss

        # Save model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_recons_loss = valid_recons_loss

            if args.variational:
                best_kl_loss = valid_kl_loss

            best_epoch = e
            torch.save(model, model_file)

        print(f'Epoch: {e + 1}')
        print('\tTraining:')
        print(f'\t\tLoss: {epoch_loss:.2f}')
        print(f'\t\tReconstruction Loss: {epoch_recons_loss:.2f}')

        if args.variational:
            print(f'\t\tKL Loss: {epoch_kl_loss:.2f}')

        print('\tValidating:')
        print(f'\t\tLoss: {valid_loss:.2f}')
        print(f'\t\tReconstruction Loss: {valid_recons_loss:.2f}')

        if args.variational:
            print(f'\t\tKL Loss: {valid_kl_loss:.2f}')

        # Logging
        log_dict = {
            'train_loss': epoch_loss,
            'train_recons_loss': epoch_recons_loss,
            'valid_loss': valid_loss,
            'valid_recons_loss': valid_recons_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': e+1,
        }

        if args.variational:
            log_dict['train_kl_loss'] = epoch_kl_loss
            log_dict['valid_kl_loss'] = valid_kl_loss

        wandb.log(log_dict)

    print('Training summary:')
    print(f'\tBest epoch: {best_epoch + 1}')
    print(f'\tBest validation loss: {best_valid_loss:.2f}')
    print(f'\tBest validation reconstruction loss: {best_recons_loss:.2f}')

    if args.variational:
        print(f'\tBest validation KL loss: {best_kl_loss:.2f}')

    print(f'\tModel saved to {model_file}')

    # Run summary
    wandb.run.summary['best_valid_loss'] = best_valid_loss
    wandb.run.summary['best_recons_loss'] = best_recons_loss

    if args.variational:
        wandb.run.summary['best_kl_loss'] = best_kl_loss

    wandb.run.summary['best_epoch'] = best_epoch + 1

    model = torch.load(model_file)

    # Testing
    model.eval()

    test_recons_loss = 0.
    num_temperatures = 0

    if args.variational:
        test_kl_loss = 0.
        num_latents = 0

    with torch.no_grad():
        for batch_grids, batch_heights in test_set:
            batch_grids = batch_grids.to(device)
            batch_heights = batch_heights.to(device)

            batch_preds = model(batch_grids)

            # Unpack batch_preds if variational mode
            if args.variational:
                batch_preds, batch_means, batch_log_vars = batch_preds

            # Compute target masks
            batch_grid_masks = heights_to_grid_masks(
                batch_preds.shape, batch_heights)

            # Compute the masked reconstruction loss
            recons_loss = reconstruction_loss(batch_preds, batch_grids)
            recons_loss = recons_loss * batch_grid_masks
            recons_loss = recons_loss.sum()

            # Compute latent masks (masks are for useless target grids)
            if args.variational:
                batch_latent_masks = heights_to_latent_masks(
                    batch_means.shape, batch_heights)

                kl = kl_loss(batch_means, batch_log_vars)
                kl = kl * batch_latent_masks
                kl = kl.sum()

            test_recons_loss += recons_loss.item()
            num_temperatures += batch_grid_masks.sum().item()

            if args.variational:
                test_kl_loss += kl.item()
                num_latents += batch_latent_masks.sum().item()

    # Normalize total reconstruction and KL losses
    test_recons_loss /= num_temperatures
    test_loss = test_recons_loss

    if args.variational:
        test_kl_loss /= num_latents
        test_loss += args.kl_weight * test_kl_loss

    print('Testing:')
    print(f'\tLoss: {test_loss:.2f}')
    print(f'\tReconstruction Loss: {test_recons_loss:.2f}')

    if args.variational:
        print(f'\tKL Loss: {test_kl_loss:.2f}')

    wandb.run.summary['test_loss'] = test_loss
    wandb.run.summary['test_recons_loss'] = test_recons_loss

    if args.variational:
        wandb.run.summary['test_kl_loss'] = test_kl_loss


if __name__ == '__main__':

    args = parse_args()

    main(args)
