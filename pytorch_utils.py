import torch


def heights_to_grid_masks(shape, heights):
    """
    Create grid masks from the input shape and heights vector.

    args:
    -----
        shape: 4-tuple of int
            The shape tuple for the corresponding masks
            shape[0]: number of resulting grid masks (num_grids)
            shape[1]: any
            shape[2]: the maximum height of the resulting grid masks (H_max)
            shape[3]: the width of the resulting grid masks (W)
        heights: tensor of size [num_grids]
            The actual height of each grid.

    returns:
    --------
        masks: tensor of size [num_grids, 1, H_max, W]
            The resulting grid masks. A grid mask for an input grid i (characterized
            by heights[i]) is such that:
            masks[i, :, 0:heights[i], :] = 1
            masks[i, :, heights[i]:H_max, :] = 0
            
            A dimension is added on the second axis as channel dimension.
    """

    device = heights.device

    # N, 1, H_max, W
    num_grids, _, max_height, width = shape

    # [1, 1, H_max, 1]
    layer_indexes = torch.arange(
        start=max_height-1, end=-1, step=-1).view(1, 1, max_height, 1)

    # [N, 1, H_max, W]
    layer_indexes = layer_indexes.expand(num_grids, 1, max_height, width)
    layer_indexes = layer_indexes.to(device)

    # [N, 1, H_max, W]
    masks = layer_indexes < heights.view(num_grids, 1, 1, 1)

    return masks


def heights_to_latent_masks(shape, heights):
    """
    Create latent masks from the input shape and heights vector.

    args:
    -----
        shape: 2-tuple of int
            The shape tuple for the corresponding masks:
            shape[0]: number of resulting latent masks (N)
            shape[1]: the size of the latent dimension (Z)
        heights: tensor of size [N]
            The actual height of each grid.

    returns:
    --------
        masks: tensor of size [N, Z]
            The resulting latent masks. A latent mask for an input latent vector i
            (characterized by heights[i]) is such that:
            masks[i] = torch.ones(Z) if heights[i] != 0, i.e. the current
                latent representation matches a grid of non-zero height equal to
                heights[i]
            masks[i] = torch.zeros(Z) if heights[i] == 0, i.e. the current
                latent representation matches a grid of zero height (artificial
                padding grid)
    """

    # N, Z
    num_grids, _ = shape

    # [N, 1]
    masks = heights.view(num_grids, 1) != 0

    return masks


def remove_temporal_padding(seq_padded_grids, seq_padded_heights):
    """
    Remove temporal padding from seq_padded_grids.

    args:
    -----
        seq_padded_grids: tensor of size [T_max, 1, H_max, W]
            A sequence of padded grids of size [1, H_max, W]. Each sequence can
            be padded temporally (along seq_padded_grids.shape[0]) and also
            spatially (along seq_padded_grids.shape[1])
        seq_padded_heights: tensor of size [T_max]
            The sequence of heights corresponding to the evolution of the sequence
            of grids in seq_padded_grids, i.e. seq_padded_heights[i] corresponds to
            the actual height of the input grid (characterized by seq_padded_grids)
            at timestep i. Whenever one encounters a zero in seq_padded_heights, say
            at index k, it means that temporal padding has started: the remaining
            grids in seq_padded_grids[k:] will be padded grids.

    returns:
    --------
        new_grids: tensor of size [k, 1, H_max, W]
            The sequence of grids from which temporal padding has been removed.
        new_heights: tensor of size [k]
            The sequence of heights from which temporal padding has been removed.
    """

    if 0 in seq_padded_heights:
        ind = (seq_padded_heights == 0).nonzero(as_tuple=True)[0][0]
        return seq_padded_grids[:ind], seq_padded_heights[:ind]

    else:
        return seq_padded_grids, seq_padded_heights
