# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/pyrovelocity/tctsti/blob/main/docs/notebooks/nsde_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="7EH5CnTDyDo9"
# Neural Stochastic Differential Equations (SDEs) model latent representations of nacent and mature mRNA kinetics

# %% [markdown] id="A5lZRKzsDF9p"
# NOTE: If running this notebook please use runtime w/ GPU
#

# %% colab={"base_uri": "https://localhost:8080/"} id="BLFGeJgFq4ve" outputId="29700bb9-5bb3-44b9-c568-00272272939d"
# %pip install torchsde
# %pip install pykeops
# %pip install scanpy
# %pip install scvelo
# %pip install anndata
# %pip install geomloss
# %pip install pytorch_lightning
# %pip install umap-learn
# %pip install timm
# %pip install igraph
# %pip install louvain

# %% colab={"base_uri": "https://localhost:8080/"} id="TkBKSD8rrC_0" outputId="7365bd09-f29e-4011-8ab6-e09ab687ceb8"
from urllib import request

import anndata
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import scvelo as scv
import torch
import torchsde
import umap
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ProgressBar, StochasticWeightAveraging
from timm.scheduler import TanhLRScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsde import BrownianInterval

# %% [markdown] id="WAki56w-rMAO"
# Our lightning model will take a a few different torch.nn.Modules as inputs. We have our SDE model (a drift + diffusion field) and an autoencoder module. Before I had the regularization model, but let's try without that and just do the drift with the "reaction kinetics" form

# %% [markdown] id="zhHl8i6PsLOx"
# We begin with the autoencoder. We will have two encoder functions $\phi_u$ and $\phi(s)$ that map unspliced and spliced counts to our "latent spliced" and "latent unspliced" features $z_u$ and $z_s$ respectively. Then there will be decoders that can map them back i.e. $ u = \phi_u(z_u)$, $s = \phi_s(z_s)$. I haven't explored different architectures for these, but maybe we can just copy whatever scVI does.


# %% id="zZsxd-wkrLR8"
class autoencoder(torch.nn.Module):
    def __init__(self, num_genes, latent_pairs):
        super().__init__()

        self.num_genes = num_genes
        self.latent_pairs = latent_pairs

        self.encoder_u = torch.nn.Sequential(
            nn.Linear(num_genes, int(num_genes / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 2), int(num_genes / 4)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 4), int(num_genes / 8)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 8), int(num_genes / 32)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 32), latent_pairs),
        )

        self.encoder_s = torch.nn.Sequential(
            nn.Linear(num_genes, int(num_genes / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 2), int(num_genes / 4)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 4), int(num_genes / 8)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 8), int(num_genes / 32)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 32), latent_pairs),
        )

        self.decoder_u = torch.nn.Sequential(
            nn.Linear(latent_pairs, int(num_genes / 32)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 32), int(num_genes / 16)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 16), int(num_genes / 8)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 8), int(num_genes / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 2), num_genes),
        )

        self.decoder_s = torch.nn.Sequential(
            nn.Linear(latent_pairs, int(num_genes / 32)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 32), int(num_genes / 16)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 16), int(num_genes / 8)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 8), int(num_genes / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(num_genes / 2), num_genes),
        )


# %% [markdown] id="GdeK1bnruB4E"
# Now this is really the core of the model, how the SDE's drift field is defined. We really want to explore what $\alpha(z_u, z_s)$ is doing, so let's define the parameterization the drift field with $\alpha(z_u, z_s)$. We can always modify *alpha_fn* to take any form, but let's start with just a deep neural network


# %% id="9YLkY1jfuAMv"
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_pairs):
        super().__init__()

        self.N = latent_pairs

        # note that the alpha doesn't care what is "unspliced" vs "spliced"
        # latent variables for now. it maps vector in 2*latent_pairs to
        # vector in 1*latent_pairs
        self.alpha_fn = torch.nn.Sequential(
            nn.Linear(self.N * 2, int(self.N * 1.5)),
            nn.Tanh(),
            nn.Linear(int(self.N * 1.5), int(self.N * 1.25)),
            nn.Tanh(),
            nn.Linear(int(self.N * 1.25), self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Softplus(),
        )

        # beta (splicing rate) will be a function of just unspliced variables
        # although maybe this isn't the right assumption?
        self.beta_fn = torch.nn.Sequential(
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Softplus(),
        )

        # gamma (degredation rate) will be a function of just unspliced variables
        self.gamma_fn = torch.nn.Sequential(
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Tanh(),
            nn.Linear(self.N, self.N),
            nn.Softplus(),
        )

        # keep sigma simple for now
        self.sigma = torch.nn.Sequential(nn.Linear(int(self.N * 2), int(self.N * 2)))

    # unregularized drift
    def f(self, t, y):
        u = y[:, : self.N]
        s = y[:, self.N :]
        drift_u = self.alpha_fn(y) - self.beta_fn(u)
        drift_s = self.beta_fn(u) - self.gamma_fn(s)

        return torch.cat((drift_u, drift_s), dim=1).float()  # shape (batch_size, state_size)

    # could we start with constant diffusion and scale it up?
    # diffusion field
    def g(self, t, y):
        return self.sigma(y).float()


# %% id="SYT1OcTtDR3y"
class light_module(pl.LightningModule):
    def __init__(self, initial_cells, SDE, autoencoder, n_integrations=100, lr=0.0001):
        super().__init__()
        cuda = torch.device("cuda")

        self.training_phase = 0
        self.autoencoder = autoencoder.to(device=cuda)
        self.sde = SDE.to(device=cuda)

        self.ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=2, backend="tensorized")
        self.softmax = nn.Softmax(dim=1)
        self.mse_loss_fn = torch.nn.MSELoss()

        self.initial_cells = torch.from_numpy(initial_cells)
        self.X0 = torch.from_numpy(initial_cells).float().to(device=cuda).float()
        self.n_integrations = n_integrations
        self.t_fwd = torch.from_numpy(np.linspace(0, 1, 60)).to(device=cuda)
        self.delta_t = float(1) / float(60)
        self.lr = lr
        self.ot_loss_history = []
        self.vae_recon_loss_history = []
        self.grad_norm_hist = []
        self.latent_pairs = autoencoder.latent_pairs
        self.n_genes = autoencoder.num_genes

        cuda = torch.device("cuda")

    def _initialize_weights(self):
        for m in self.sde.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.autoencoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def training_step(self, batch, batch_idx):
        # in the first training phase we can just fit the autoencoder
        if self.training_phase == 0:
            cuda = torch.device("cuda")

            U, S = batch[:, 0 : self.n_genes], batch[:, self.n_genes :]
            Z_U, Z_S = self.autoencoder.encoder_u(U), self.autoencoder.encoder_s(S)
            X_U, X_S = self.autoencoder.decoder_u(Z_U), self.autoencoder.decoder_s(Z_S)
            X_recon = torch.cat((X_U, X_S), dim=1)
            loss = self.mse_loss_fn(X_recon, batch) * self.n_genes

            self.log("pretraining_loss", loss, on_step=True, prog_bar=True)

            return loss

        # in the second training phase we learn the SDE
        if self.training_phase != 0:
            X0_samp_idxs = np.random.choice(self.X0.shape[0], self.n_integrations)
            self.X0_samp = self.X0[X0_samp_idxs, :]

            cuda = torch.device("cuda")

            U, S = batch[:, 0 : self.n_genes], batch[:, self.n_genes :]
            Z_U, Z_S = self.autoencoder.encoder_u(U), self.autoencoder.encoder_s(S)
            X_U, X_S = self.autoencoder.decoder_u(Z_U), self.autoencoder.decoder_s(Z_S)
            X_recon = torch.cat((X_U, X_S), dim=1)
            vae_recon_loss = self.mse_loss_fn(X_recon, batch) * self.n_genes

            X0_ = self.X0_samp
            U0, S0 = X0_[:, : self.n_genes], X0_[:, self.n_genes :]
            Z_U0, Z_S0 = (
                self.autoencoder.encoder_u(U0).to(device=cuda).float(),
                self.autoencoder.encoder_s(S0).to(device=cuda).float(),
            )

            U0_recon, S0_recon = self.autoencoder.decoder_u(Z_U0), self.autoencoder.decoder_s(Z_S0)
            X0_recon = torch.cat((U0_recon, S0_recon), dim=1)
            initial_pos_loss = self.mse_loss_fn(X0_recon, X0_) * self.n_genes

            Z0_ = torch.cat((Z_U0, Z_S0), dim=1)
            self.Z0_ = Z0_

            Z_pred = torchsde.sdeint(self.sde, Z0_, self.t_fwd, method="euler")

            Z_pred_flat = torch.flatten(Z_pred, start_dim=0, end_dim=1)

            U_fwd, S_fwd = Z_pred_flat[:, : self.latent_pairs], Z_pred_flat[:, self.latent_pairs :]
            X_U_fwd, X_S_fwd = self.autoencoder.decoder_u(U_fwd), self.autoencoder.decoder_s(S_fwd)
            X_fwd = torch.cat((X_U_fwd, X_S_fwd), dim=1)

            ot_loss = self.ot_loss_fn(X_fwd, batch)
            loss = ot_loss + vae_recon_loss + initial_pos_loss

            self.ot_loss_history.append(ot_loss.cpu().detach().numpy())
            self.vae_recon_loss_history.append(vae_recon_loss.cpu().detach().numpy())
            self.log("transport_loss", ot_loss, on_step=True, prog_bar=True)
            self.log("recon_loss", vae_recon_loss, on_step=True, prog_bar=True)
            self.log("initial_loss", initial_pos_loss, on_step=True, prog_bar=True)

            return loss

    def configure_optimizers(self):
        parameters = [{"params": self.parameters(), "lr": self.lr}]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        return optimizer

    def on_after_backward(self):
        # Custom gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

        # Now compute and log the global grad norm
        grad_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item() ** 2
                grad_norm += param_norm

        grad_norm = grad_norm ** (1.0 / 2)
        self.grad_norm_hist.append(grad_norm)


# %% id="Q78c4HC0C4G8"
# dataset class
class dataset_(Dataset):
    def __init__(self, X):
        self.cuda = torch.device("cuda")
        self.X = torch.from_numpy(X).float().to(device=self.cuda)

    def __len__(self):
        return self.X.size(dim=0)

    def __getitem__(self, idx):
        return self.X[idx, :]


# %% colab={"base_uri": "https://localhost:8080/", "height": 178, "referenced_widgets": ["def080c9195c4a3c805d6a264c450947", "6e37b7bea5c54ee087ab2402481a628c", "13b818505c0340379c304e8c026ab406", "d13fa56acd1045a8a4200dd38c180005", "080c32c2995f4277bfcaa4ffc300164f", "c3e987519c6a4036b98e48bd6104c7d2", "6e2ef709d3124a6bbf11c9a22cce49ee", "028657347e5946a1a873161303a35840", "fed2e5b599574b808d4098e9a2b9cb5b", "91112e293d824863893926ea1451261d", "116f03bf2aa74a73ad923badf96b113a"]} id="Va0RyUy3HF8k" outputId="24a30d82-183d-4f65-e627-1590104b775f"
# load the pancreas dataset
adata = scv.datasets.pancreas()
adata

# %% colab={"base_uri": "https://localhost:8080/"} id="jWxRlcOgN-ry" outputId="8841360f-7890-4962-d68c-42df0e4952f1"
scv.pp.normalize_per_cell(adata, counts_per_cell_after=1000)  ### I need to think about the preprocessing a lot more.
scv.pp.filter_genes_dispersion(adata, n_top_genes=3000)

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="ss1rcwQ2OG0r" outputId="01e88e05-b72d-420c-e82f-3b6b534f0433"
scv.pl.umap(adata)

# %% [markdown] id="8FPGIchNH9Py"
# I need to choose an initial population, let's choose the cells with the highest S_score in the cycling progenitors. (ideally I'd like to have an option to automate the selection of initial cells, although I think using our own knowledge of the biological system is important in real applications)

# %% colab={"base_uri": "https://localhost:8080/", "height": 482} id="T_WMB7RmIilV" outputId="325fea82-a0c1-4812-cbec-b125a16de87f"
scv.pl.umap(adata, color="S_score")

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="2SF8YTVAI4dt" outputId="2bc65d34-d541-4c42-fc74-7f01b52b3551"
adata.obs["initial_cells"] = (adata.obs["S_score"] >= adata.obs["S_score"].quantile(0.95)) & (
    adata.obs["clusters"] == "Ductal"
)
scv.pl.umap(adata, color="initial_cells")

# %% id="g0j83p0bEkWu"
X = np.concatenate((adata.layers["unspliced"].todense(), adata.layers["spliced"].todense()), axis=1)
dataset = dataset_(X)
batch_size = 2000
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
X0 = np.concatenate(
    (
        adata.layers["unspliced"][adata.obs["initial_cells"] == "True", :].todense(),
        adata.layers["spliced"][adata.obs["initial_cells"] == "True", :].todense(),
    ),
    axis=1,
)

# %% id="bvIE4yolwhLe"
# n_latent_variables = 30

# sde = SDE(n_latent_variables)
# module.sde = sde.to(device="cuda")

# %% colab={"base_uri": "https://localhost:8080/"} id="xuWqIXIdNv5Z" outputId="01e2ed70-09df-41fb-fa14-c5cc87e43af8"
n_latent_variables = 30

sde = SDE(n_latent_variables)
ae = autoencoder(3000, n_latent_variables)

module = light_module(X0, sde, ae, n_integrations=100, lr=0.0001)
module.training_phase = 0
trainer = pl.Trainer(
    callbacks=[StochasticWeightAveraging(swa_lrs=1e-3)],
    accumulate_grad_batches=1,
    max_epochs=300,
    enable_progress_bar=True,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 436, "referenced_widgets": ["6ab3fb34d5564956ac54c68219695352", "0bd0882aac294058ac50c6ec1256c1ac", "cb2c5c4d7a164b58a150808809435733", "d379ef6fe2d745babc830e9c3747771a", "4d71526d51eb4fca834c2bd6e8b9cd2d", "1637ede9456544518a8865ab6ab3e3b7", "eb3c77fa32a54ec6a3c562c12a7e0075", "569f44831e6d49c6a7dbe2e4132800a3", "3282ec8999a3463a91fb743395b2be4f", "b33e4dc9fa7945619ec39c44fbe42e2e", "cb67f1baf6c64fbba360875e3dd665de"]} id="0u5P24mBDkl8" outputId="94626be5-9ebc-4a11-e053-96b08f3cb991"
module.lr = 0.001  # learning rate can be high for the VAE pretraining relative to the learning rate for the full model
module.training_phase = 0
trainer.fit(module, dataloader)

# %% colab={"base_uri": "https://localhost:8080/", "height": 398, "referenced_widgets": ["f7dd1f1d7e264132ba8825e6c09cb720", "5bc12b733c8f4346aa4227256588cadf", "88950944058b4d028fab69e38b5b19d3", "ab42ee5b5f804a8f929c99e49e784c5c", "1cba6caa9aa145448f178d507fe73717", "2993003fd10e4303b1141071d65152d0", "2a41679a45d3439da525a534e19f4adf", "b1fd9e57f3344b60bcffbb7fd5ad69e7", "897dbaf8a6a642269bb494f321f4b74c", "bbec6dcd25f343af9237c5a8f7e63e00", "25c518a27573454fbedf9fdeec0765bd"]} id="TpcoUuLTCdtp" outputId="0f9de2ae-eaef-43be-d16c-bd8cbae6f803"
trainer = pl.Trainer(max_epochs=300, enable_progress_bar=True)
module.lr = 0.00005
module.training_phase = 1
trainer.fit(module, dataloader)

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="CpYvcDPaEAY4" outputId="bc7a4a84-2ecb-4bce-ff15-025b3f25a145"
plt.plot(module.ot_loss_history, label="OT Loss")
plt.plot(module.vae_recon_loss_history, label="MSE Recon Loss")

plt.legend()
plt.show()

# %% id="igQMSg2WxuD1"
