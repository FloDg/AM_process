# AM_process
Machine learning-based approach for temperature field prediction during the additive manufacturing process. Our approach combines a variational autoencoder (VAE) that models the process in space with a recurrent neural network (RNN) that models the process in time inside a reduced latent space. A description of our VAE-RNN approach is available [here](ARIAC_GD1_2D_grid_material_addition.pdf). 

We provide code for training the VAE and RNN models, and for vizualizing the predictions of the trained models.

# Dataset

To use the code, first download the dataset and save it in a folder called "data/". Please note that the dataset is not available online. To get access to it, send an email to caroline.sainvitu@cenaero.be. The dataset consists of 2D simulations of additive manufacturing processes with an evolving domain. You can learn more by referring to the `utils.py` file.

The code requires three libraries to be installed: torch, wandb, and matplotlib.

## Training the VAE

To train the VAE, run the following command:

```python train_vae.py --help```

This will display detailed parameters that you can use to configure the training. If you'd like to test the workflow with a smaller dataset, you can run:

```python train_vae.py --num_simulations 15 ```

This will use 15 simulations instead of the entire dataset.

## Training the latent simulator

To train the latent simulator, use the following command:

```python train_latentsimulator.py --help```

This will show you the available parameters that you can use. If you want to test the workflow with a smaller dataset, you can run:

```python train_latentsimulator.py --vae_file path_to_vae_model --num_simulations 15 ```

Here, `path_to_vae_model` should be replaced with the path to your previously trained VAE.

## Creating a GIF

To create a GIF to visualize your experiment results, you can use the `plot_simulation.ipynb` notebook. Note that you will need to modify the path to the models in the notebook.