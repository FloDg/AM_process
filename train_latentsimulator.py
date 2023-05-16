import os
import wandb
import torch
import argparse

from math import inf
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import create_dataset
from pytorch_utils import heights_to_grid_masks
from vae import reconstruction_loss
from latentsimulator import LatentSimulator


def load_dataset(sequence_stride, batch_size, encoder, num_simulations=inf):
    """
    Load a dataset arranged for the latent space simulator.

    args:
    -----
        sequence_stride: int
            The sequence stride with which sequences should be under-sampled, along
            the temporal axis.
        batch_size: int
            The number of sequences per batch.
        encoder: nn.Module instance
            The encoder model to generate the latent representations of input
            sequences.
        num_simulations: int
            The number of simulation sequences to load into the dataset, from the
            entire set of sequences.

    returns:
    --------
        train_set: DataLoader instance
            The training DataLoader instance for training a latent simulator.
            Each element of a batch is a tuple of (features, windows, grids, heights),
            where:
            features is a tensor of size [T_max, n_features] representing the input
                features (laser power, break time, ...) for the given sequence
            windows is a tensor of size [T_max, latent_dim] representing
                the encoded time windows for the given sequence
            grids is a tensor of size
                [T_max, H_max, W]
                representing the (encoded) grid of temperatures
            heights is a tensor of size [T_max] corresponding to the actual
                height of the (encoded) grid.
        valid_set: DataLoader instance
            The validation DataLoader instance for training a latent simulator.
            Each element of a batch is a tuple of (features, grids, heights),
            where:
            features is a tensor of size [T_max, n_features] representing the input
                features (laser power, break time, ...)
            grids is a tensor of size
                [T_max, H_max, W]
                representing the (encoded) grid of temperatures
            heights is a tensor of size [T_max] corresponding to the actual
                height of the (encoded) grid.
        test_set: DataLoader instance
            The testing DataLoader instance for training a latent simulator.
            Each element of a batch is a tuple of (features, grids, heights),
            where:
            features is a tensor of size [T_max, n_features] representing the input
                features (laser power, break time, ...)
            grids is a tensor of size
                [T_max, H_max, W]
                representing the (encoded) grid of temperatures
            heights is a tensor of size [T_max] corresponding to the actual
                height of the (encoded) grid.
        initial_latent: tensor of size [latent_dim]
            The initial encoded grid of temperatures
        mean_latent_grids: tensor of size [latent_dim]
            The tensor of mean encoded temperatures, for the training set.
        std_latent_grids: tensor of size [latent_dim]
            The tensor of std encoded temperatures, for the training set.
        mean_features: tensor of size [n_features]
            The tensor of mean input features, for the training set.
        std_features: tensor of size [n_features]
            The tensor of std input features, for the training set.
        max_length: int
            The maximum length of any sequence in the training set (T_max).
        max_height: int
            The maximum height reached by any sequence in the training set (H_max).
        width: int
            The width of the grids.
    """

    window_size = 1

    # Create train, val and test datasets with appropriate stride and window size.
    # Since we are training a RNN, we are interested in grid sequences.
    dataset = create_dataset(sequence_stride=sequence_stride,
                             window_size=window_size,
                             use_transitions=False,
                             num_simulations=num_simulations)

    # Transform grids into tensors and move them to device
    train_grids = torch.tensor(dataset['train_seq_grids']).float()
    del dataset['train_seq_grids']

    valid_grids = torch.tensor(dataset['valid_seq_grids']).float()
    del dataset['valid_seq_grids']

    test_grids = torch.tensor(dataset['test_seq_grids']).float()
    del dataset['test_seq_grids']

    # Transform heights into tensors and move them to device
    train_heights = torch.tensor(
        dataset['train_seq_heights']).float()
    del dataset['train_seq_heights']

    valid_heights = torch.tensor(
        dataset['valid_seq_heights']).float()
    del dataset['valid_seq_heights']

    test_heights = torch.tensor(dataset['test_seq_heights'])
    del dataset['test_seq_heights']

    # Transform features into tensors and move them to device
    train_features = torch.tensor(
        dataset['train_seq_features']).float()
    del dataset['train_seq_features']

    valid_features = torch.tensor(
        dataset['valid_seq_features']).float()
    del dataset['valid_seq_features']

    test_features = torch.tensor(
        dataset['test_seq_features']).float()
    del dataset['test_seq_features']

    # Transform windows into tensors and move them to device
    train_windows = torch.tensor(
        dataset['train_seq_windows']).float()
    del dataset['train_seq_windows']

    _, max_length, max_height, width = train_grids.shape

    # Add channel dimension to training windows as they have to be encoded
    # Merge batch, temporal and window_size dimensions to feed to encoder
    train_windows = train_windows.view(-1, 1, max_height, width)

    # Encode training grids and windows
    with torch.no_grad():
        encoder.eval()
        train_encoded_windows = encoder.encode(train_windows)  # [N*T, latent]

    latent_dim = train_encoded_windows.shape[-1]

    # Compute mean and std of encoded grids of temperatures and input features

    with torch.no_grad():
        std_latent_grids, mean_latent_grids = torch.std_mean(
            encoder.encode(train_grids.view(-1, 1, max_height, width)), dim=0)
    std_features, mean_features = torch.std_mean(train_features, dim=(0, 1))

    # Recover temporal dimension in encoded windows
    train_encoded_windows = train_encoded_windows.view(
        -1, max_length, latent_dim)

    # Create the DataLoader instances
    # train_features: [N, T_max, num_features]
    # train_encoded_windows: currently???? [N, T_max, wsize, latent]
    # train_grids: if encoded: [N, T_max, latent] else: [N, T_max, H_max, W]
    # train_heights: [N, T_max]
    train_set = DataLoader(TensorDataset(train_features, train_encoded_windows,
                                         train_grids, train_heights),
                           batch_size=batch_size, shuffle=True)

    valid_set = DataLoader(TensorDataset(valid_features, valid_grids, valid_heights),
                           batch_size=batch_size, shuffle=True)

    test_set = DataLoader(TensorDataset(test_features, test_grids, test_heights),
                          batch_size=batch_size, shuffle=True)

    # Define initial grid temperatures
    initial_grid = torch.full((1, 1, max_height, width), 20.)
    with torch.no_grad():
        initial_latent = encoder.encode(initial_grid).squeeze()

    return (train_set, valid_set, test_set, initial_latent,
            mean_latent_grids, std_latent_grids, mean_features, std_features,
            max_length, max_height, width)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--num_simulations', type=int, default=inf,
                        help='Number of simulations to load from the '
                             'dataset (for debug purposes), default is all '
                             'simulations.')
    parser.add_argument('--sequence_stride', type=int, default=10,
                        help='Sequence stride for the dataset, default is 10.')
    parser.add_argument('--normalize_features', action='store_true',
                        help='Whether to normalize the laser input features '
                             'or not.')
    parser.add_argument('--normalize_latents', action='store_true',
                        help='Whether to normalize the latent inputs.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model, default is 10.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size, default is 64.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate, default is 0.001.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Whether to use GPU or not.')
    parser.add_argument('--lr_scheduling', action='store_true',
                        help='Whether to use learning rate scheduling or not.')

    # Model parameters
    parser.add_argument('--num_recs', type=int, default=1,
                        help='Number of recurrent layers, default is 1.')
    parser.add_argument('--hidden_states', type=int, default=2048,
                        help='Number of hidden states per recurrent layer, '
                             'default is 2048')
    parser.add_argument('--num_fcs', type=int, default=1,
                        help='Number of fully connected layers, default is 1.')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Number of hidden units of the fully connected '
                             'layers, default is 2048.')
    parser.add_argument('--vae_file', type=str, required=True,
                        help='Path to the pretrained VAE.')
    parser.add_argument('--model_dir', type=str,
                        default='models/latentsimulator',
                        help='Directory where the model will be saved, '
                        'default is "models/latentsimulator".')
    args = parser.parse_args()

    # Initialize wandb project
    wandb.init(project='latentsimulator', config=args)
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

    # Autoencoder file
    vae_file = args.vae_file

    # Where to save the model
    if wandb.run.name is None:
        filename = 'model.pt'

    else:
        filename = wandb.run.name + '.pt'

    model_file = os.path.join(args.model_dir, filename)

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Load encoder
    encoder = torch.load(vae_file, map_location='cpu')
    encoder.eval()

    # Load train, valid and test dataset
    (train_set, valid_set, test_set, initial_latent,
     mean_latents, std_latents, mean_features, std_features,
     max_length, max_height, width) = load_dataset(
        sequence_stride=args.sequence_stride,
        batch_size=args.batch_size,
        encoder=encoder,
        num_simulations=args.num_simulations)

    # Move initial encoded grid to device
    initial_latent = initial_latent.to(device)

    batch_features, batch_encoded_windows, _, _ = next(iter(train_set))
    _, max_length, features_size = batch_features.shape
    *_, latent_dim = batch_encoded_windows.shape

    if not args.normalize_features:
        mean_features, std_features = None, None

    if not args.normalize_latents:
        mean_latents, std_latents = None, None

    # Initialize model
    model = LatentSimulator(
        latent_dim=latent_dim, input_size=features_size,
        num_recs=args.num_recs, hidden_states=args.hidden_states,
        num_fcs=args.num_fcs, hidden_size=args.hidden_size,
        mean_latents=mean_latents, std_latents=std_latents,
        mean_features=mean_features, std_features=std_features
    )

    # Send models to device
    encoder.to(device)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.lr_scheduling:
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # Training loop
    best_epoch = 0
    best_valid_loss = inf
    for e in range(args.epochs):
        model.train()

        epoch_recons_loss = 0.
        num_temperatures = 0

        for batch_features, _, batch_grids, batch_heights in train_set:

            # Move to device
            batch_features = batch_features.to(device)
            batch_grids = batch_grids.to(device)
            batch_heights = batch_heights.to(device)

            batch_size, seq_length, _ = batch_features.shape

            batch_preds = model.simulate(
                batch_features,
                initial_latent.expand(batch_grids.shape[0], latent_dim))

            batch_preds = encoder.decode(
                batch_preds.view(-1, latent_dim))  # [B*T_max, H_max, W]

            batch_heights = batch_heights.view(-1)  # [B*T_max]
            # [B*T_max, 1, H_max, W]
            batch_grids = batch_grids.view(-1, 1, *batch_grids.shape[2:])

            batch_grid_masks = heights_to_grid_masks(
                batch_grids.shape, batch_heights)  # [B*T_max, 1, H_max, W]

            batch_preds = batch_preds.view(
                batch_grids.shape)  # [B*T_max, 1, H_max, W]

            recons_loss = reconstruction_loss(batch_preds, batch_grids)
            recons_loss = recons_loss * batch_grid_masks
            recons_loss = recons_loss.sum()

            loss = recons_loss / batch_grid_masks.sum()

            epoch_recons_loss += recons_loss.item()
            num_temperatures += batch_grid_masks.sum().item()

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.lr_scheduling:
            scheduler.step()

        epoch_recons_loss /= num_temperatures
        epoch_loss = epoch_recons_loss

        # Validation phase
        model.eval()

        valid_recons_loss = 0.
        num_temperatures = 0

        with torch.no_grad():
            for batch_features, batch_grids, batch_heights in valid_set:

                # Move to device
                batch_features = batch_features.to(device)
                batch_grids = batch_grids.to(device)
                batch_heights = batch_heights.to(device)

                batch_size, seq_length, _ = batch_features.shape

                batch_preds = model.simulate(
                    batch_features,
                    initial_latent.expand(batch_grids.shape[0], latent_dim))

                batch_preds = encoder.decode(
                    batch_preds.view(-1, latent_dim))  # [B*T_max, H_max, W]

                batch_heights = batch_heights.view(-1)  # [B*T_max]
                # [B*T_max, 1, H_max, W]
                batch_grids = batch_grids.view(-1, 1, *batch_grids.shape[2:])

                batch_grid_masks = heights_to_grid_masks(
                    batch_grids.shape, batch_heights)  # [B*T_max, 1, H_max, W]

                batch_preds = batch_preds.view(
                    batch_grids.shape)  # [B*T_max, 1, H_max, W]

                recons_loss = reconstruction_loss(batch_preds, batch_grids)
                recons_loss = recons_loss * batch_grid_masks
                recons_loss = recons_loss.sum()

                valid_recons_loss += recons_loss.item()
                num_temperatures += batch_grid_masks.sum().item()

        valid_recons_loss /= num_temperatures
        valid_loss = valid_recons_loss

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_recons_loss = valid_recons_loss

            best_epoch = e
            torch.save(model, model_file)

        print(f'Epoch: {e + 1}')
        print('\tTraining:')
        print(f'\t\tLoss: {epoch_loss:.2f}')
        print(f'\t\tReconstruction loss: {epoch_recons_loss:.2f}')
        print('\tValidating:')
        print(f'\t\tLoss: {valid_loss:.2f}')
        print(f'\t\ttReconstruction Loss: {valid_recons_loss:.2f}')
        print(flush=True)

        # Logging
        log_dict = {
            'train_loss': epoch_loss,
            'train_recons_loss': epoch_recons_loss,
            'valid_loss': valid_loss,
            'valid_recons_loss': valid_recons_loss,
            'epoch': e+1,
        }

        wandb.log(log_dict)

    print('Training summary:')
    print(f'\tBest epoch: {best_epoch + 1}')
    print(f'\tBest validation loss: {best_valid_loss:.2f}')
    print(f'\tBest validation reconstruction loss: {best_recons_loss:.2f}')
    print(f'\tModel saved to {model_file}')
    print(flush=True)

    # Run summary
    wandb.run.summary['best_valid_loss'] = best_valid_loss
    wandb.run.summary['best_recons_loss'] = best_recons_loss
    wandb.run.summary['best_epoch'] = best_epoch + 1

    model = torch.load(model_file)

    # Testing
    model.eval()
    test_recons_loss = 0.
    num_temperatures = 0

    with torch.no_grad():
        for batch_features, batch_grids, batch_heights in valid_set:

            # Move to device
            batch_features = batch_features.to(device)
            batch_grids = batch_grids.to(device)
            batch_heights = batch_heights.to(device)

            batch_size, seq_length, _ = batch_features.shape

            batch_preds = model.simulate(
                batch_features,
                initial_latent.expand(batch_grids.shape[0], latent_dim))

            batch_preds = encoder.decode(
                batch_preds.view(-1, latent_dim))  # [B*T_max, H_max, W]

            batch_heights = batch_heights.view(-1)  # [B*T_max]
            # [B*T_max, 1, H_max, W]
            batch_grids = batch_grids.view(-1, 1, *batch_grids.shape[2:])

            batch_grid_masks = heights_to_grid_masks(
                batch_grids.shape, batch_heights)  # [B*T_max, 1, H_max, W]

            batch_preds = batch_preds.view(
                batch_grids.shape)  # [B*T_max, 1, H_max, W]

            recons_loss = reconstruction_loss(batch_preds, batch_grids)
            recons_loss = recons_loss * batch_grid_masks
            recons_loss = recons_loss.sum()

            test_recons_loss += recons_loss.item()
            num_temperatures += batch_grid_masks.sum().item()

    test_recons_loss /= num_temperatures
    test_loss = test_recons_loss

    print('Testing:')
    print(f'\tLoss: {test_loss:.2f}')
    print(f'\tReconstruction loss: {test_recons_loss:.2f}')

    wandb.run.summary['test_loss'] = test_loss
    wandb.run.summary['test_recons_loss'] = test_recons_loss
