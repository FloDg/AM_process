import torch
from torch import nn, distributions
from math import ceil, floor

# Convolutional layers constants
PADDING = 1
STRIDE = 2
DILATION = 1
KERNEL_SIZE = 3


def Conv(in_channels, out_channels, kernel_size=KERNEL_SIZE,
         stride=STRIDE, padding=PADDING, dilation=DILATION):
    """
    Creates a new convolutional layer used in the encoder. It is composed of a
    convolutional layer, a batch normalization layer and a LeakyReLU activation.

    args:
    -----
        in_channels: int
            Number of channels of the input.
        out_channels: int
            Number of channels of the output.
        kernel_size: int (3)
            Size of the convolutional kernel.
        stride: int (2)
            Stride of the convolution.
        padding: int (1)
            Padding of the convolution.
        dilation: int (1)
            Dilation of the convolution.

    returns:
    --------
        nn.Sequential
            The convolutional layer.
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, padding=padding,
                  stride=stride, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    ).float()


def TransConv(in_channels, out_channels, add_padding,
              kernel_size=KERNEL_SIZE, stride=STRIDE,
              padding=PADDING, dilation=DILATION):
    """
    Creates a new transposed convolutional layer used in the decoder. It is
    composed of a transposed convolutional layer, a batch normalization layer
    and a LeakyReLU activation.

    args:
    -----
        in_channels: int
            Number of channels of the input.
        out_channels: int
            Number of channels of the output.
        add_padding: bool
            Whether to add a padding of 1 to the output.
        kernel_size: int (3)
            Size of the convolutional kernel.
        stride: int (2)
            Stride of the convolution.
        padding: int (1)
            Padding of the convolution.
        dilation: int (1)
            Dilation of the convolution.

    returns:
    --------
        nn.Sequential
            The transposed convolutional layer.
    """

    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation,
                           output_padding=1 if add_padding else 0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


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


def reparametrize(means, log_vars):
    """
    Reparametrization trick to sample from a Gaussian distribution.

    args:
    -----
        means: tensor of shape [batch_size, latent_dim]
            Means of the Gaussian distributions.
        log_vars: tensor of shape [batch_size, latent_dim]
            Logarithm of the variances of the Gaussian distributions.

    returns:
    --------
        tensor of shape [batch_size, latent_dim]
            Samples from the Gaussian distributions.
    """

    stds = torch.exp(log_vars / 2)
    q = distributions.Normal(means, stds)
    return q.rsample()


def reconstruction_loss(x_hat, x):
    """
    Reconstruction loss, i.e. the mean squared error between the input and the
    reconstruction.

    args:
    -----
        x_hat: tensor of shape [batch_size, channels, height, width]
            Reconstructed images.
        x: tensor of shape [batch_size, channels, height, width]
            Input images.

    returns:
    --------
        tensor of shape [batch_size]
            Reconstruction losses of each image.
    """

    return nn.functional.mse_loss(x_hat, x, reduction='none')


def kl_loss(means, log_vars):
    """
    KL divergence loss between the Gaussian distributions and the standard
    normal distribution. Used to train the variational autoencoder.

    args:
    -----
        means: tensor of shape [batch_size, latent_dim]
            Means of the Gaussian distributions.
        log_vars: tensor of shape [batch_size, latent_dim]
            Logarithm of the variances of the Gaussian distributions.

    returns:
    --------
        tensor of shape [batch_size]
            KL divergence losses of each latent image.
    """

    return torch.sum(-0.5 * (1 + log_vars - means ** 2 - log_vars.exp()), dim=1)


def compute_out_shapes(input_shape, num_convs, k=KERNEL_SIZE,
                       p=PADDING, s=STRIDE, d=DILATION):
    """
    Computes the output shapes of the transposed convolutional layers of the
    decoder, to match the output shapes of the encoder convolutional layers.

    args:
    -----
        input_shape: 2-tuple of int
            Shape of the input images.
            input_shape[0]: height
            input_shape[1]: width
        num_convs: int
            Number of convolutional layers.
        k: int (3)
            Size of the convolutional kernel.
        p: int (1)
            Padding of the convolution.
        s: int (2)
            Stride of the convolution.
        d: int (1)
            Dilation of the convolution.

    returns:
    --------
        list of 2-tuples of int
            List of output shapes of the transposed convolutional layers.
            list[i][0]: height of the i-th output shape
            list[i][1]: width of the i-th output shape
    """

    shapes = []
    H_curr, W_curr = input_shape
    for _ in range(num_convs):
        H_curr = floor((H_curr + 2 * p - d * (k - 1) - 1)/s + 1)
        W_curr = floor((W_curr + 2 * p - d * (k - 1) - 1)/s + 1)
        shapes.append((H_curr, W_curr))
    return shapes


class Encoder(nn.Module):
    """Encoder module"""

    def __init__(self,
                 input_channels, input_shape,
                 latent_dim,
                 num_convs, hidden_channels,
                 num_fcs, hidden_size,
                 variational=False):
        """
        Creates a new encoder module.

        args:
        -----
            input_channels: int
                Number of channels of the input images.
            input_shape: 2-tuple of int
                Shape of the input images.
                shape[0]: height
                shape[1]: width
            latent_dim: int
                Dimension of the latent space.
            num_convs: int
                Number of convolutional layers.
            hidden_channels: int
                Number of channels of the first convolutional layer.
                The i-th convolutional layer will have
                'hidden_channels * (2 ** (i - 1))' channels.
            num_fcs: int
                Number of fully connected layers.
            hidden_size: int
                Number of neurons of the fully connected layers.
            variational: bool (False)
                Whether to use a variational encoder or not.
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_convs = num_convs
        self.hidden_channels = hidden_channels
        self.num_fcs = num_fcs
        self.hidden_size = hidden_size
        self.variational = variational

        # Convolutional layers
        self.convs = [Conv(input_channels, hidden_channels)] \
            + [Conv(hidden_channels * (2 ** i), hidden_channels * (2 ** (i+1)))
               for i in range(num_convs - 1)]

        self.convs = nn.Sequential(*self.convs)

        # Final dimensions
        self.final_shape = (ceil(input_shape[0] / (2 ** num_convs)),
                            ceil(input_shape[1] / (2 ** num_convs)))

        self.final_channels = 2 ** (num_convs-1) * hidden_channels

        self.flattened_size = self.final_channels * \
            self.final_shape[0] * self.final_shape[1]

        # Flattening layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fcs = [FC(self.flattened_size, hidden_size)] \
            + [FC(hidden_size, hidden_size)
               for _ in range(num_fcs - 1)]

        self.fcs = nn.Sequential(*self.fcs)

        # Output layers for means (and log variances if variational)
        self.mean_fc = nn.Linear(hidden_size, latent_dim)

        if variational:
            self.var_fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        Forward pass of the encoder. Encode the input images into latent
        representations. If the encoder is variational, returns the means and
        log variances of the Gaussian distributions. Otherwise, returns encoded
        images.

        args:
        -----
            x: tensor of shape [batch_size, input_channels, height, width]
                Input images.

        returns:
        --------
            if variational:
                tensor of shape [batch_size, latent_dim]
                    Means of the Gaussian distributions.
                tensor of shape [batch_size, latent_dim]
                    Log variances of the Gaussian distributions.

            else:
                tensor of shape [batch_size, latent_dim]
                    Encoded images.
        """

        # First go through the convolutional layers
        x = self.convs(x)

        # Then flatten
        x = self.flatten(x)

        # Then go through the fully connected layers
        x = self.fcs(x)

        # Finally compute the outputs
        means = self.mean_fc(x)

        if self.variational:
            log_vars = self.var_fc(x)
            return means, log_vars

        else:
            return means


class Decoder(nn.Module):
    """Decoder module"""

    def __init__(self,
                 latent_dim, flattened_size,
                 input_channels, output_shape,
                 output_channels,
                 num_transconvs,
                 num_fcs, hidden_size):
        """
        Creates a new decoder module.

        args:
        -----
            latent_dim: int
                Dimension of the latent space.
            flattened_size: int
                Size of the last fully-connected layer output.
            input_channels: int
                Number of channels of the input of the first transposed
                convolutional layer.
            output_shape: 2-tuple of int
                Shape of the reconstructed images.
                shape[0]: height
                shape[1]: width
            output_channels: int
                Number of channels of the reconstructed images.
            num_transconvs: int
                Number of transposed convolutional layers.
            num_fcs: int
                Number of fully connected layers.
            hidden_size: int
                Number of neurons of the hidden fully connected layers.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.flattened_size = flattened_size
        self.input_channels = input_channels
        self.output_shape = output_shape
        self.output_channels = output_channels
        self.num_transconvs = num_transconvs
        self.num_fcs = num_fcs
        self.hidden_size = hidden_size

        # Fully connected layers
        if num_fcs > 1:
            self.fcs = [FC(latent_dim, hidden_size)] \
                + [FC(hidden_size, hidden_size)
                   for _ in range(num_fcs - 2)] \
                + [FC(hidden_size, flattened_size)]

        else:
            self.fcs = [FC(latent_dim, flattened_size)]

        self.fcs = nn.Sequential(*self.fcs)
        self.shapes = [output_shape] + \
            compute_out_shapes(output_shape, num_transconvs)

        # Unflattening layer
        self.unflatten = nn.Unflatten(1, (input_channels, *self.shapes[-1]))

        # Transposed convolutional layers
        self.transconvs = [TransConv(input_channels // (2 ** i),
                                     input_channels // (2 ** (i+1)),
                                     not (self.shapes[-(2 + i)][0] % 2))
                           for i in range(num_transconvs)]

        self.transconvs = nn.Sequential(*self.transconvs)

        final_channels = input_channels // (2 ** num_transconvs)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(final_channels, output_channels,
                                    kernel_size=KERNEL_SIZE, padding=PADDING)

    def forward(self, x):
        """
        Forward pass of the decoder. Decode the latent representations into
        images.

        args:
        -----
            x: tensor of shape [batch_size, latent_dim]
                Latent representations.

        returns:
        --------
            tensor of shape [batch_size, output_channels, height, width]
                Reconstructed images.
        """

        # First go through the fully connected layers
        x = self.fcs(x)

        # Then unflatten
        x = self.unflatten(x)

        # Then go through the transposed convolutional layers
        x = self.transconvs(x)

        # Finally go through the convolutional layer
        x = self.final_conv(x)
        return x


class AutoEncoder(nn.Module):
    """Auto-encoder model"""

    def __init__(self,
                 input_channels, input_shape,
                 latent_dim,
                 num_convs, hidden_channels,
                 num_fcs, hidden_size,
                 mean_inputs=None, std_inputs=None,
                 variational=False):
        """
        Creates a new auto-encoder model.

        args:
        -----
            input_channels: int
                Number of channels of the input images.
            input_shape: 2-tuple of int
                Shape of the input images.
                shape[0]: height
                shape[1]: width
            latent_dim: int
                Dimension of the latent space.
            num_convs: int
                Number of convolutional layers.
            hidden_channels: int
                Number of channels of the first convolutional layer of the encoder.
                The i-th convolutional layer will have
                'hidden_channels * (2 ** (i - 1))' channels.
            num_fcs: int
                Number of fully connected layers.
            hidden_size: int
                Number of neurons of the fully connected layers.
            mean_inputs: None | tensor of shape [input_channels, height, width] (None)
                Mean of the inputs (set to None if no normalization).
            std_inputs: None | tensor of shape [input_channels, height, width] (None)
                Standard deviation of the inputs (set to None if no normalization).
            variational: bool (False)
                Whether to use a variational auto-encoder.
        """

        super().__init__()

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_convs = num_convs
        self.hidden_channels = hidden_channels
        self.num_fcs = num_fcs
        self.hidden_size = hidden_size
        self.variational = variational

        self.normalize = mean_inputs != None and std_inputs != None
        if self.normalize:
            self.register_buffer('mean_inputs', mean_inputs)
            self.register_buffer('std_inputs', std_inputs)

        self.encoder = Encoder(
            input_channels=input_channels, input_shape=input_shape,
            latent_dim=latent_dim,
            num_convs=num_convs, hidden_channels=hidden_channels,
            num_fcs=num_fcs, hidden_size=hidden_size,
            variational=variational
        )

        self.decoder = Decoder(
            latent_dim=latent_dim, flattened_size=self.encoder.flattened_size,
            input_channels=self.encoder.final_channels,
            output_shape=input_shape,
            output_channels=input_channels,
            num_transconvs=num_convs,
            num_fcs=num_fcs, hidden_size=hidden_size
        )

    def forward(self, x):
        """
        Encode and decode the inputs. Used for training. Returns the
        reconstructed inputs. If the model is variational, also returns the
        means and the logarithms of the variances of the latent variables.

        args:
        -----
            x: tensor of shape [batch_size, input_channels, height, width]
                Input images.

        returns:
        --------
            tensor of shape [batch_size, input_channels, height, width]
                Reconstructed images.

            if variational:
                means: tensor of shape [batch_size, latent_dim]
                    Means of the latent variables.
                log_vars: tensor of shape [batch_size, latent_dim]
                    Logarithms of the variances of the latent variables.
        """

        if self.normalize:
            x = (x - self.mean_inputs) / self.std_inputs

        if self.variational:
            means, log_vars = self.encoder(x)
            z = reparametrize(means, log_vars)

            x_hat = self.decoder(z)

            if self.normalize:
                x_hat = x_hat * self.std_inputs + self.mean_inputs

            return x_hat, means, log_vars

        else:
            z = self.encoder(x)

            x_hat = self.decoder(z)

            if self.normalize:
                x_hat = x_hat * self.std_inputs + self.mean_inputs

            return x_hat

    def encode(self, x):
        """
        Encode a batch of images.

        args:
        -----
            x: tensor of shape [batch_size, input_channels, height, width]
                Input images.

        returns:
        --------
            tensor of shape [batch_size, latent_dim]
                Encoded images.
        """
        if self.normalize:
            x = (x - self.mean_inputs) / self.std_inputs

        if self.variational:
            # Return the mean latent vector
            z, _ = self.encoder(x)

        else:
            z = self.encoder(x)

        return z

    def decode(self, z):
        """
        Decode a batch of latent vectors.

        args:
        -----
            z: tensor of shape [batch_size, latent_dim]
                Input latent vectors.

        returns:
        --------
            tensor of shape [batch_size, input_channels, height, width]
                Decoded images.
        """
        x = self.decoder(z)

        if self.normalize:
            x = x * self.std_inputs + self.mean_inputs

        return x
