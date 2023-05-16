import torch
from torch import nn


def RecurrentLayers(input_size, hidden_states, num_layers):
    """
    Creates a new multi-layer GRU.

    args:
    -----
        input_size: int
            Dimension of the input features.
        hidden_states: int
            Size of hidden state in the recurrent layers.
        num_layers: int
            Number of recurrent layers.

    returns:
    --------
        nn.GRU
            A new multi-layer GRU.
    """

    return nn.GRU(input_size, hidden_states, num_layers,
                  batch_first=True)


def FC(in_features, out_features):
    """
    Creates a new fully connected layer. It is composed of a linear layer and a
    LeakyReLU activation.

    args:
    -----
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.

    returns:
    --------
        nn.Sequential
            The fully connected layer.
    """

    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True)
    )


class TimeDistributed(nn.Module):
    """
    Wraps a module to apply it to each time step of a sequence.
    """

    def __init__(self, submodule):
        """
        Creates a new TimeDistributed module.

        args:
        -----
            submodule: nn.Module
                The module to apply to each time step.

        returns:
        --------
            TimeDistributed
                A new TimeDistributed module.
        """

        super().__init__()

        self.submodule = submodule

    def forward(self, x):
        """
        Applies the module to each time step of the sequence.

        args:
        -----
            x: tensor of shape [batch_size, seq_length, ...]

        returns:
        --------
            tensor of shape [batch_size, seq_length, ...]
                The output of the module.
        """
        batch_size, seq_length, _ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)

        x = self.submodule(x)
        x = x.unflatten(0, (batch_size, seq_length))

        return x


class LatentSimulator(nn.Module):
    """Latent simulator model"""

    def __init__(self,
                 latent_dim, input_size,
                 num_recs, hidden_states,
                 num_fcs, hidden_size,
                 mean_latents=None, std_latents=None,
                 mean_features=None, std_features=None):
        """
        Create a latent simulator model.

        args:
        -----
            latent_dim: int
                Dimension of the latent space.
            input_size: int
                Dimension of the input features.
            num_recs: int
                Number of recurrent layers.
            hidden_states: int
                Number of hidden states in the recurrent layers.
            num_fcs: int
                Number of fully connected layers.
            hidden_size: int
                Number of hidden units in the fully connected layers.
            mean_latents: None | tensor of shape [latent_dim] (None)
                Mean of the latent vectors. If None, the model will not normalize
                the latent vectors.
            std_latents: None | tensor of shape [latent_dim] (None)
                Standard deviation of the latent vectors. If None, the model will
                not normalize the latent vectors.
            mean_features: None | tensor of shape [input_size] (None)
                Mean of the input features. If None, the model will not normalize
                the input features.
            std_features: None | tensor of shape [input_size] (None)
                Standard deviation of the input features. If None, the model will
                not normalize the input features.
        """

        super().__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_recs = num_recs
        self.hidden_states = hidden_states
        self.num_fcs = num_fcs
        self.hidden_size = hidden_size

        self.normalize_latents = mean_latents != None and std_latents != None
        self.normalize_features = mean_features != None and std_features != None

        if self.normalize_latents:
            self.register_buffer('mean_latents', mean_latents)
            self.register_buffer('std_latents', std_latents)

        if self.normalize_features:
            self.register_buffer('mean_features', mean_features)
            self.register_buffer('std_features', std_features)

        # Recurrent layers
        self.recs = RecurrentLayers(
            latent_dim + input_size, hidden_states, num_recs)

        # Fully connected layers
        self.fcs = [FC(hidden_states, hidden_size)] \
            + [FC(hidden_size, hidden_size)
               for _ in range(num_fcs - 1)]

        self.fcs = nn.Sequential(*self.fcs)

        # Wrap the FC layers inside a TimeDistributed layer
        self.fcs = TimeDistributed(self.fcs)

        # Outputs layers
        # Output is a vector with a length of `latent_dim`
        self.output_fc = nn.Linear(hidden_size, latent_dim)
        self.output_fc = TimeDistributed(self.output_fc)

    def forward(self, features, latents, hidden_states=None):
        """
        Predict the next latent vectors using the real latent vectors.

        args:
        -----
            features: tensor of shape [batch_size, seq_length, input_size]
                Input features.
            latents: tensor of shape [batch_size, seq_length, latent_dim]
                Real latent vectors.
            hidden_states: tensor of shape [num_recs, batch_size, hidden_states] (None)
                Initial hidden states of the recurrent layers. If None, the
                hidden states will be initialized to zeros.

        returns:
        --------
        tensor of shape [batch_size, seq_length, latent_dim]
            Predicted latent vectors.
        tensor of shape [num_recs, batch_size, hidden_states]
            Final hidden states of the recurrent layers.
        """

        if self.normalize_features:
            features = (features - self.mean_features) / self.std_features

        if self.normalize_latents:
            latents = (latents - self.mean_latents) / self.std_latents

        # First concat current features to previous latent state
        x = torch.cat((features, latents), dim=2)

        # Then go through the recurrent layers
        x, hidden_states = self.recs(x, hidden_states)

        # Then go through the fully connected layers
        x = self.fcs(x)

        # Finally compute the outputs
        new_latents = self.output_fc(x)

        if self.normalize_latents:
            new_latents = new_latents * self.std_latents + self.mean_latents

        return new_latents, hidden_states

    def simulate(self, seq_features, initial_latent, hidden_states=None):
        """
        Simulate the next latent vectors using the model, starting from an
        initial latent.

        args:
        -----
            features: tensor of shape [batch_size, seq_length, input_size]
                Input features.
            initial_latent: tensor of shape [batch_size, latent_dim]
                Initial latent vector.
            hidden_states: None | tensor of shape [num_recs, batch_size, hidden_states] (None)
                Initial hidden states of the recurrent layers. If None, the
                hidden states will be initialized to zeros.

        returns:
        --------
            tensor of shape [batch_size, seq_length, latent_dim]
                Predicted latent vectors.
        """

        batch_size, seq_length, _ = seq_features.shape
        device = seq_features.device

        prev_latent = initial_latent.unsqueeze(1)  # [B, 1, latent]

        if self.normalize_latents:
            prev_latent = (prev_latent - self.mean_latents) / self.std_latents

        if self.normalize_features:
            seq_features = (seq_features - self.mean_features) / \
                self.std_features

        seq_preds = torch.empty(batch_size, seq_length,
                                self.latent_dim, device=device)  # [B, T_max, latent]

        for t in range(seq_length):
            features = seq_features[:, t, :].unsqueeze(1)  # [B, 1, n_features]

            preds, hidden_states = self(
                features, prev_latent, hidden_states)  # [B, 1, latent], [B, hidden_size]

            prev_latent = preds

            seq_preds[:, t, :] = preds.squeeze(1)

        if self.normalize_latents:
            seq_preds = seq_preds * self.std_latents + self.mean_latents

        return seq_preds
