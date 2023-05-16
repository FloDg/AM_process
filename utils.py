import os
import math
import numpy as np

from random import Random

SIMULATION_FILE = 'data/3HWpqvRBy-{}/npz_data/data.npz'
MINAMO_FILE = 'data/3HWpqvRBy-{}/Minamo_Parameters-Wall2D.txt'


def create_masks(shape, heights):
    """
    Generate the masks with a given shape from a given array of heights.

    args:
    -----
        shape: tuple with the shape of the masks, [N, T_max, H_max, W]
        heights: 2D numpy array of heights, formatted as follows:
            - 1st dim identifies the simulation
            - 2nd dim identifies the timestep
            - heights[i, t] is the number of layers that the material
              had at timestep t in the simulation i (useful for spatial
              padding)
            - if heights[i, t] is 0, it means that the simulation i has
              ended before timestep t (useful for temporal padding)

    returns:
    --------
        masks: Boolean array of shape [N, T_max, H_max, W]
            If masks[i, t, h, w] is True, than the temperature
            at (h, w) at timestep t of simulation i should be
            taken into account in the computation of the
            loss/metric
    """

    # N, T_max, H_max, W
    num_sequences, max_length, max_height, width = shape

    # [1, 1, H_max]
    ys = np.arange(start=max_height-1, stop=-1, step=-1).reshape(1, 1, -1)

    # [N, T_max, H_max]
    ys = np.tile(ys, (num_sequences, max_length, 1))

    # [N, T_max, H_max]
    masks = ys < np.expand_dims(heights, axis=-1)

    # [N, T_max, H_max, 1]
    masks = np.expand_dims(masks, axis=-1)

    # [N, T_max, H_max, W]
    masks = np.tile(masks, (1, 1, 1, width))

    return masks


def masked_mse(preds, targets, heights):
    """
    args:
    -----
        preds: numpy array with predicted grids, [N, T_max, H_max, W]
        targets: numpy array with target grids, [N, T_max, H_max, W]
        heights: numpy array, [N, T_max]
            The array of the heights reached by the piece,
            at each timestep, for each sequence
            (If 0, then the end of simulation is reached)

    return:
    -------
        The (temporally and spatially) masked MSE
    """

    masks = create_masks(preds.shape, heights)

    # [N, T_max, H_max, W]
    masked_difference = (preds - targets) * masks

    # Scalar
    return (masked_difference ** 2).sum() / (masks.sum())


def load_simulation(simulation_id):
    """
    args:
    -----
        simulation_id: integer

    return:
    -------
        data: the simulation data
        static_params: the static parameters (P and b) of the simulation
    """

    npz_file = SIMULATION_FILE.format(simulation_id)
    data = np.load(npz_file, allow_pickle=True)

    with open(MINAMO_FILE.format(simulation_id), 'r') as f:
        static_params = [float(line.split()[2]) for line in f.readlines()]

    return data, static_params


def load_all_features_and_grids(add_static_params, sequence_stride,
                                num_simulations=math.inf,
                                use_absolute_time=False):
    """
    args:
    -----
        add_static_params: boolean
            Whether or not to include the static parameters P and b in the
            feature vector
        sequence_stride: integer
        num_simulations: integer

    return:
    -------
        batch_seq_features: list of sequences represented by numpy arrays.
            Each sequence represents a simulation and is a 2D array
            of shape (T, num_features).  The features are delta_t, laser_x,
            laser_y (and static_param if static_param = True)
        batch_seq_grids: list of sequences of length T, where each sequence is
            composed of 2D grids containing the temperatures, where each grid
            can have a different height.
    """

    batch_seq_features, batch_seq_grids = [], []

    simulation_id = 1
    while simulation_id <= num_simulations:
        if not os.path.exists(SIMULATION_FILE.format(simulation_id)):
            break

        data, (nominal_power, break_time) = load_simulation(simulation_id)

        # Extract input data: `t`, `x_t`, `P_t`
        seq_time = data['time'].astype(float)
        seq_laser_x = data['laser_position_x'].astype(float)
        seq_laser_y = data['laser_position_y'].astype(float)
        seq_laser_power = data['laser_power'].astype(float)

        if add_static_params:
            # Create features `P` and `b`
            seq_nominal_power = np.full(seq_laser_power.shape, nominal_power)
            seq_break_time = np.full(seq_laser_power.shape, break_time)

        if use_absolute_time:
            seq_time_feature = seq_time
        else:
            # Create a feature `delta`
            seq_time_feature = seq_time.copy()
            seq_time_feature[1:] = seq_time[1:] - seq_time[:-1]

        # If static parameters, add them to the features
        static_params = (
            seq_nominal_power, seq_break_time) if add_static_params else ()

        # Load the temperature grid
        seq_features = np.stack(
            (seq_time_feature, seq_laser_x, seq_laser_y, seq_laser_power,
             *static_params), axis=1)

        seq_grids = data['temperatures']

        # Stride the sequences
        seq_features = seq_features[::sequence_stride]
        seq_grids = seq_grids[::sequence_stride]
        seq_grids = list(map(lambda x: x.astype(float), seq_grids))

        batch_seq_features.append(seq_features)
        batch_seq_grids.append(seq_grids)

        simulation_id += 1

    return batch_seq_features, batch_seq_grids


def get_heights(seq_grids):
    return list(map(lambda seq_grid: seq_grid.shape[0], seq_grids))


def pad_grids(seq_grids, max_height, default_temperature):
    padded_grids = np.array(list(map(lambda x: np.concatenate((np.full(
        (max_height - x.shape[0], x.shape[1]), default_temperature), x), axis=0), seq_grids)))
    return padded_grids


def create_windows(batch_seq_grids, window_size, default_temperature):
    """
    args:
    -----
        batch_seq_grids: list of numpy arrays with dim [T, H, W],
            The array of sequences of grids.
        window_size: integer, size of the window
        default_temperature: float, default temperature used in the initial
            window

    return:
    -------
        windows: list of numpy arrays with dim [T, window_size, H, W]
            The list of sequences of windows associated to each sequence of grids.
    """

    batch_seq_windows = []

    # Spatial dimensions for the grid
    spatial_dims = batch_seq_grids[0].shape[-2:]

    # Initial window is full of default temperatures, [window_size, H, W]
    window = np.full((window_size, *spatial_dims), default_temperature)

    # Iterate over each sequence
    for seq_grids in batch_seq_grids:
        seq_windows = np.zeros((len(seq_grids), window_size,
                                *spatial_dims))

        # Iterate over each time step
        for t, grid in enumerate(seq_grids):
            seq_windows[t] = window

            # Update the window
            grid = grid.reshape(1, *spatial_dims)
            window = np.concatenate((window[1:], grid), axis=0)

        batch_seq_windows.append(seq_windows)

    return batch_seq_windows


def create_dataset(sequence_stride,
                   window_size, default_temperature=20.,
                   validation_size=.1, test_size=.1,
                   add_static_params=False, num_simulations=math.inf,
                   use_absolute_time=False, use_transitions=False,
                   return_list=False,
                   seed=0):
    """
    args:
    -----
        sequence_stride: integer, the step between two timesteps to
            load. If sequence_stride is 10, 1 out of 10 timesteps is kept
        window_size: integer, the size of the window
        default_temperature: float (default=20.), the default temperature
            used in the initial window
        validation_size: float (default=0.1), the ratio of sequences used
            for validation.
        test_size: float (default=0.1), the ratio of sequences used
            for testing.
        add_static_params: boolean (default=False), whether or not to add
            the static parameters of the simulation to the features
        num_simulations: integer (default=math.inf), number of simulations
            to load
        use_absolute_time: boolean (default=False), whether or not to use
            absolute time
        use_transitions: boolean (default=False), whether or not to use
            transitions in the train set. If set to True, the two first
            dimensions (N & T_max) of the arrays in the train set are merged
            together.
        seed: integer, the seed (should not be modified)

    return:
    -------
        dataset: a dictionnary containing all the simulations.
            - dataset["train_seq_features"] is a numpy array of shape
              [N, T_max, num_features]. It contains all the features
              i.e., the laser position, power, etc
            - dataset["train_seq_grids"] is a numpy array of shape
              [N, T_max, H_max, W]. It contains all the padded temperature
              grids. These are the targets
            - dataset["train_seq_heights"] is a numpy array of shape
            [N, T_max]. It allows to compute the masks.

            The same apply for valid and test fields.

            If window_size > 0:
                - dataset["train_seq_windows"] is a numpy array of shape
                [N, T_max, window_size, H_max, W]. It contains the windows
                used to predict the next grid.

            For a given simulation i and a given timestep t,
            - the inputs are:
                - dataset["train_seq_features"][i, t]
                - dataset["train_seq_windows"][i, t]
            - the targets is dataset["train_seq_grids"][i, t]
            - the list of heights to compute the mask is
              dataset["train_seq_heights"][i, t]

    """

    batch_seq_features, batch_seq_grids = load_all_features_and_grids(
        add_static_params, sequence_stride, num_simulations, use_absolute_time)

    if return_list:
        return batch_seq_features, batch_seq_grids

    # First find maximum number of layers (among all simulations and timesteps)
    max_height = max(
        map(lambda seq_grids: seq_grids[-1].shape[0], batch_seq_grids))

    # Then retrieve the evolution of the heights
    batch_seq_heights = list(
        map(lambda seq_grids: get_heights(seq_grids), batch_seq_grids))

    # Then (spatially) pad all grids
    batch_seq_grids = list(map(lambda seq_grids: pad_grids(
        seq_grids, max_height, default_temperature), batch_seq_grids))

    # Then find maximum sequence length
    max_length = max(map(lambda seq_grids: len(seq_grids), batch_seq_grids))

    # Split in 3 sets
    n_sequences = len(batch_seq_features)
    train_size = 1 - validation_size - test_size
    train_end = int(np.round(train_size * n_sequences))
    valid_end = int(np.round(validation_size * n_sequences))

    Random(seed).shuffle(batch_seq_features)
    Random(seed).shuffle(batch_seq_grids)
    Random(seed).shuffle(batch_seq_heights)

    # Split train set from the two other sets
    train_seq_features = batch_seq_features[:train_end]
    train_seq_grids = batch_seq_grids[:train_end]
    train_seq_heights = batch_seq_heights[:train_end]

    validtest_seq_features = batch_seq_features[train_end:]
    validtest_seq_grids = batch_seq_grids[train_end:]
    validtest_seq_heights = batch_seq_heights[train_end:]

    # Then (temporally) pad all sequences of valid and test sets
    validtest_seq_features = list(map(lambda seq_features: np.concatenate((seq_features, np.zeros(
        (max_length - seq_features.shape[0], *seq_features.shape[1:]))), axis=0), validtest_seq_features))

    validtest_seq_grids = list(map(lambda seq_grids: np.concatenate((seq_grids, np.full(
        (max_length - seq_grids.shape[0], *seq_grids.shape[1:]), default_temperature)), axis=0), validtest_seq_grids))

    validtest_seq_heights = list(map(lambda seq_heights: seq_heights + (max_length - len(seq_heights)) * [0],
                                     validtest_seq_heights))

    # Turn them into numpy arrays
    validtest_seq_features = np.array(validtest_seq_features)
    validtest_seq_grids = np.array(validtest_seq_grids)
    validtest_seq_heights = np.array(validtest_seq_heights)

    # For the train set, create the windows if necessary
    if window_size > 0:
        train_seq_windows = create_windows(
            train_seq_grids, window_size, default_temperature)

    # If use_transitions, merge the two first dimensions (N and T)
    if use_transitions:
        train_seq_features = np.concatenate(train_seq_features, axis=0)
        train_seq_grids = np.concatenate(train_seq_grids, axis=0)
        train_seq_heights = np.concatenate(train_seq_heights, axis=0)

        if window_size > 0:
            train_seq_windows = np.concatenate(train_seq_windows, axis=0)

    # Else, pad (temporally) the sequences
    else:
        train_seq_features = list(map(lambda seq_features: np.concatenate((seq_features, np.zeros(
            (max_length - seq_features.shape[0], *seq_features.shape[1:]))), axis=0), train_seq_features))

        train_seq_grids = list(map(lambda seq_grids: np.concatenate((seq_grids, np.full(
            (max_length - seq_grids.shape[0], *seq_grids.shape[1:]), default_temperature)), axis=0), train_seq_grids))

        train_seq_heights = list(map(lambda seq_heights: seq_heights + (max_length - len(seq_heights)) * [0],
                                     train_seq_heights))

        if window_size > 0:
            train_seq_windows = list(map(lambda seq_windows: np.concatenate((seq_windows, np.full(
                (max_length - seq_windows.shape[0], *seq_windows.shape[1:]), default_temperature)), axis=0), train_seq_windows))

        train_seq_features = np.array(train_seq_features)
        train_seq_grids = np.array(train_seq_grids)
        train_seq_heights = np.array(train_seq_heights)

        if window_size > 0:
            train_seq_windows = np.array(train_seq_windows)
    
    # Create the dictionnary
    dataset = {}

    dataset['train_seq_features'] = train_seq_features
    dataset['train_seq_grids'] = train_seq_grids
    dataset['train_seq_heights'] = train_seq_heights

    if window_size > 0:
        dataset['train_seq_windows'] = train_seq_windows

    dataset['valid_seq_features'] = validtest_seq_features[:valid_end]
    dataset['valid_seq_grids'] = validtest_seq_grids[:valid_end]
    dataset['valid_seq_heights'] = validtest_seq_heights[:valid_end]

    dataset['test_seq_features'] = validtest_seq_features[valid_end:]
    dataset['test_seq_grids'] = validtest_seq_grids[valid_end:]
    dataset['test_seq_heights'] = validtest_seq_heights[valid_end:]

    return dataset


if __name__ == '__main__':
    # Train set contains sequences
    print('----- Train set contains sequences -----')
    dataset = create_dataset(sequence_stride=10, window_size=2,
                             default_temperature=20., num_simulations=10,
                             validation_size=.1, test_size=.1,
                             add_static_params=True, use_absolute_time=False,
                             use_transitions=False)

    print("dataset['train_seq_features']", dataset['train_seq_features'].shape)
    print("dataset['train_seq_grids']", dataset['train_seq_grids'].shape)
    print("dataset['train_seq_heights']", dataset['train_seq_heights'].shape)
    print("dataset['train_seq_windows']", dataset["train_seq_windows"].shape)
    print("dataset['valid_seq_features']", dataset['valid_seq_features'].shape)
    print("dataset['valid_seq_grids']", dataset['valid_seq_grids'].shape)
    print("dataset['valid_seq_heights']", dataset['valid_seq_heights'].shape)
    print("dataset['test_seq_features']", dataset['test_seq_features'].shape)
    print("dataset['test_seq_grids']", dataset['test_seq_grids'].shape)
    print("dataset['test_seq_heights']", dataset['test_seq_heights'].shape)

    X = dataset['train_seq_features'][0:2]
    X_seq = dataset['train_seq_windows'][0:2]
    y = dataset['train_seq_grids'][0:2]
    heights = dataset['train_seq_heights'][0:2]

    print('Batch of features:', X.shape)
    print('Batch of windows:', X_seq.shape)
    print('Batch of targets:', y.shape)
    print('Batch of heights:', heights.shape)

    print("MSE preds == targets", masked_mse(
        preds=y, targets=y, heights=heights))
    print("MSE preds != targets", masked_mse(
        preds=y, targets=y+1, heights=heights))

    # Train set contains transitions
    print('----- Train set contains transitions -----')
    dataset = create_dataset(sequence_stride=10, window_size=2,
                             default_temperature=20., num_simulations=10,
                             validation_size=.1, test_size=.1,
                             add_static_params=True, use_absolute_time=False,
                             use_transitions=True)

    print("dataset['train_seq_features']", dataset['train_seq_features'].shape)
    print("dataset['train_seq_grids']", dataset['train_seq_grids'].shape)
    print("dataset['train_seq_heights']", dataset['train_seq_heights'].shape)
    print("dataset['train_seq_windows']", dataset["train_seq_windows"].shape)
    print("dataset['valid_seq_features']", dataset['valid_seq_features'].shape)
    print("dataset['valid_seq_grids']", dataset['valid_seq_grids'].shape)
    print("dataset['valid_seq_heights']", dataset['valid_seq_heights'].shape)
    print("dataset['test_seq_features']", dataset['test_seq_features'].shape)
    print("dataset['test_seq_grids']", dataset['test_seq_grids'].shape)
    print("dataset['test_seq_heights']", dataset['test_seq_heights'].shape)

    X = dataset['train_seq_features'][13:17]
    X_seq = dataset['train_seq_windows'][13:17]
    y = dataset['train_seq_grids'][13:17]
    heights = dataset['train_seq_heights'][13:17]

    print('Batch of features:', X.shape)
    print('Batch of windows:', X_seq.shape)
    print('Batch of targets:', y.shape)
    print('Batch of heights:', heights.shape)
