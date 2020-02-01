import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.simple_gp import ExactGPModel
from training.training_gp import train_exact_gp
from util.visualization.plotting import default_fig_style, default_plt_style

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    np.random.seed(42)

    # plot styling
    fig = plt.figure()
    default_plt_style(plt)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=6)

    # demo function to draw from
    f = lambda x: x + np.sin(2 * (x + 0.1)) + 0.1
    dx = np.arange(0.5, 2.5, 0.001)
    dy = f(dx)

    # train gp to show epistemic uncertainty
    obs_epis_x = np.array([0.67, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.4])
    obs_epis_y = f(obs_epis_x)

    xt = torch.Tensor(obs_epis_x).to(device)
    yt = torch.Tensor(obs_epis_y).to(device)
    dxt = torch.Tensor(dx).to(device)
    dyt = torch.Tensor(dy).to(device)

    learning_rate = 0.1
    num_epochs = 110

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(xt, yt, likelihood).to(device)

    train_exact_gp(model, likelihood, xt, yt, epochs=num_epochs,
                   learning_rate=learning_rate)

    model.eval()
    with torch.no_grad():
        pred = model(dxt)
        pred_mean = pred.mean.cpu()
        pred_var = pred.variance.cpu()
        pred_std = np.sqrt(pred_var)

    ax = plt.subplot(2, 1, 1)
    ax.set_title('Epistemic Uncertainty')
    plot(obs_epis_x, obs_epis_y, pred_mean, pred_std, dx, dy, ax)

    # Homoscedastic aleotoric
    obs_aleo_x = np.arange(0.6, 2.5, 0.3)
    noise_aleo_h_std = 0.3
    obs_aleo_h_y = np.random.normal(f(obs_aleo_x), noise_aleo_h_std, [14, obs_aleo_x.size]).transpose()
    aleo_h_stds = np.repeat(noise_aleo_h_std, len(dx))

    ax = plt.subplot(2, 2, 3)
    ax.set_title('Aleatoric Homoscedastic Uncertainty')
    plot(obs_aleo_x, obs_aleo_h_y, dy, aleo_h_stds, dx, dy, ax)

    # Heteroscedastic Aleororic
    aleo_het_stds_f = lambda x: 1/ 3 * (np.sin(1.5 * x) + 1)
    obs_aleo_het_y = np.random.normal(f(obs_aleo_x), aleo_het_stds_f(obs_aleo_x), [14, obs_aleo_x.size]).transpose()
    aleo_het_stds = aleo_het_stds_f(dx)

    ax = plt.subplot(2, 2, 4)
    ax.set_title('Aleatoric Heteroscedastic Uncertainty')
    plot(obs_aleo_x, obs_aleo_het_y, dy, aleo_het_stds, dx, dy, ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def plot(obs_x, obs_y, pred_means, pred_stds, dx, dy, ax):
    # some styling
    default_fig_style(ax)
    ax.margins(0, 0.06)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])

    ax.plot(dx, dy, label='Reference Function')
    if len(obs_y.shape) > 1:
        sec_dim = obs_y.shape[1]
        obs_y = obs_y.reshape([-1])
        obs_x = np.repeat(obs_x, sec_dim)
    ax.plot(obs_x, obs_y, 'o', color='orange', label='Observations')
    # ax.plot(dx, pred_mean)
    for j in range(1, 5):
        unc = ax.fill_between(dx, pred_means - j / 2 * pred_stds, pred_means + j / 2 * pred_stds,
                        alpha=0.2, color='lightblue')
    unc.set_label('Uncertainty')


main()
