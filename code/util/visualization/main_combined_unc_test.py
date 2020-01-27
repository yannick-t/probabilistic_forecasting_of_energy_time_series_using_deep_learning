import torch
import numpy as np
import matplotlib.pyplot as plt

from models.deep_gp import DeepGaussianProcess
from models.skorch_wrappers.deep_gp_skorch import DeepGPSkorch

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')
torch.set_default_tensor_type('torch.' + ('cuda.' if use_cuda else '') + 'DoubleTensor')

f = lambda x: 1 / 8 * (np.sin(2 * (x + 0.1)) + 4)
dx = np.arange(0.5, 4.5, 0.001)
dx_test = np.arange(0.5, 4.5, 0.05)
dy = f(dx)

obs_aleo_x = np.arange(0.6, 2.5, 0.1)

aleo_het_stds_f = lambda x: 1 / 30 * (np.sin(1.5 * x) + 1.2)
obs_aleo_het_y = np.random.normal(f(obs_aleo_x), aleo_het_stds_f(obs_aleo_x), [40, obs_aleo_x.size]).transpose()
aleo_het_stds = aleo_het_stds_f(dx)

# train model on observation
sec_dim = obs_aleo_het_y.shape[1]
y_train = np.expand_dims(obs_aleo_het_y.reshape([-1]), 1)
x_train = np.expand_dims(np.repeat(obs_aleo_x, sec_dim), 1)
x_test = np.expand_dims(dx_test, 1)


# some code to test the combined uncertainty estimates of the models
# on a toy problem and visualizing results
def main():
    deep_gp()


def deep_gp():
    dgp = DeepGPSkorch(module=DeepGaussianProcess,
                       module__input_size=x_train.shape[-1],
                       module__hidden_size=[2],
                       module__output_size=y_train.shape[-1] * 2,
                       module__num_inducing=128,
                       lr=0.01,
                       max_epochs=100,
                       optimizer=torch.optim.Adam,
                       num_data=x_train.shape[0],
                       device=device)

    dgp.fit(x_train, y_train)

    pred = dgp.predict(x_test)
    pred_mean = pred[..., 0]
    pred_var = pred[..., 1]
    pred_epis_var = pred[..., 2]
    pred_aleo_var = pred[..., 3]

    # plot data
    ax = plt.subplot(2, 2, 1)
    ax.set_title('aleatoric')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, np.sqrt(pred_aleo_var), dx, dy, ax)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('epistemic')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, np.sqrt(pred_epis_var), dx, dy, ax)
    ax = plt.subplot(2, 1, 2)
    ax.set_title('combined')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, np.sqrt(pred_var), dx, dy, ax)
    plt.show()


def plot(obs_x, obs_y, pred_x, pred_means, pred_stds, dx, dy, ax):
    pred_means = pred_means.squeeze()
    pred_stds = pred_stds.squeeze()

    # some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0, 0.06)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])

    ax.plot(dx, dy, label='Reference Function')
    if len(obs_y.shape) > 1:
        sec_dim = obs_y.shape[1]
        obs_y = obs_y.reshape([-1])
        obs_x = np.repeat(obs_x, sec_dim)
    ax.plot(obs_x, obs_y, 'o', color='orange', label='Observations')
    ax.plot(pred_x, pred_means)
    for j in range(1, 5):
        unc = ax.fill_between(pred_x, pred_means - j / 2 * pred_stds, pred_means + j / 2 * pred_stds,
                        alpha=0.2, color='lightblue')

    for j in range(1, 5):
        unc = ax.fill_between(dx, dy - j / 2 * aleo_het_stds, dy + j / 2 * aleo_het_stds,
                        alpha=0.2, color='orange')


main()
