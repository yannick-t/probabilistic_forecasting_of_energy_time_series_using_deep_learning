import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_single import DeepEnsembleSingle
from models.deep_ensemble_sklearn import DeepEnsemble
from models.deep_gp import DeepGaussianProcess
from models.functional_np import RegressionFNP
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from models.skorch_wrappers.bnn_skorch import BNNSkorch
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from models.skorch_wrappers.deep_gp_skorch import DeepGPSkorch
from models.skorch_wrappers.functional_np_skorch import RegressionFNPSkorch
from models.torch_bnn import TorchBNN
from training.loss.heteroscedastic_loss import HeteroscedasticLoss

'''
some code to test the combined uncertainty estimates of the models
on a toy problem and visualizing results
'''

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

f = lambda x: 1 / 8 * (np.sin(2 * (x + 0.1)) + 4)
dx = np.arange(0.5, 4.5, 0.001)
dx_test = np.arange(0.5, 4.5, 0.05)
dy = f(dx)

obs_aleo_x = np.arange(0.6, 2.5, 0.1)

aleo_het_stds_f = lambda x: 1 * (np.sin(1.5 * x) + 1.2)
obs_aleo_het_y = np.random.normal(f(obs_aleo_x), aleo_het_stds_f(obs_aleo_x), [40, obs_aleo_x.size]).transpose()
aleo_het_stds = aleo_het_stds_f(dx)

# train model on observation
sec_dim = obs_aleo_het_y.shape[1]
y_train = np.expand_dims(obs_aleo_het_y.reshape([-1]), 1)
x_train = np.expand_dims(np.repeat(obs_aleo_x, sec_dim), 1)
x_test = np.expand_dims(dx_test, 1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
y_train = scaler.transform(y_train)
x_test = scaler.transform(x_test)



def main():
    bayesian_nn()


def bayesian_nn():
    bnn = BNNSkorch(module=TorchBNN,
                    module__input_size=x_train.shape[-1],
                    module__output_size=y_train.shape[-1] * 2,
                    module__hidden_size=[16],
                    module__prior_mu=0,
                    module__prior_sigma=0.6,
                    sample_count=30,
                    lr=0.001,
                    max_epochs=1000,
                    train_split=None,
                    batch_size=1024,
                    optimizer=torch.optim.Adam,
                    criterion=HeteroscedasticLoss,
                    device=device)

    bnn.fit(x_train, y_train)

    pred = bnn.predict(x_test)
    pred_mean = pred[..., 0]
    pred_var = pred[..., 1]
    pred_epis_std = pred[..., 2]
    pred_aleo_std = pred[..., 3]

    plot_all(pred_mean, pred_var, pred_epis_std, pred_aleo_std)


def simple_nn():
    nn = BaseNNSkorch(
        module=DeepEnsembleSingle,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=[128],
        lr=0.001,
        max_epochs=200,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        train_split=None,
        verbose=1
    )

    nn.fit(x_train, y_train)

    pred = nn.predict(x_test)
    pred_mean = pred[..., 0]
    softplus = torch.nn.Softplus()
    pred_std = softplus(torch.as_tensor(pred[..., 1])).cpu().numpy()

    plot_single(pred_mean, pred_std)


def deep_ensemble():
    deep_ens = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        hidden_size=[32],
        lr=0.001,
        max_epochs=1000,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device
    )

    deep_ens.fit(x_train, y_train)

    pred = deep_ens.predict(x_test)
    pred_mean = pred[..., 0]
    pred_var = pred[..., 1]
    pred_epis_std = pred[..., 2]
    pred_aleo_std = pred[..., 3]

    plot_all(pred_mean, pred_var, pred_epis_std, pred_aleo_std)


def functional_np():
    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        module__hidden_size=[20],
        module__dim_u=3,
        module__dim_z=50,
        module__fb_z=1.0,
        optimizer=torch.optim.Adam,
        device=device,
        max_epochs=200,
        batch_size=1024,
        reference_set_size=300,
        train_size=x_train.size)

    fnp.fit(x_train, y_train)

    pred_mean, pred_std = fnp.predict(x_test)

    # plot data
    ax = plt.subplot(1, 1, 1)
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, pred_std, dx, dy, ax)
    plt.show()


def concrete_dropout():

    conc = ConcreteSkorch(
        module=ConcreteDropoutNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=[32],
        lengthscale=1e-4,
        dataset_size=x_train.shape[0],
        sample_count=30,
        lr=0.001,
        max_epochs=3000,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        verbose=1
    )
    conc.fit(x_train, y_train)

    pred = conc.predict(x_test)
    pred_mean = pred[..., 0]
    pred_var = pred[..., 1]
    pred_epis_std = pred[..., 2]
    pred_aleo_std = pred[..., 3]

    plot_all(pred_mean, pred_var, pred_epis_std, pred_aleo_std)


def deep_gp():
    dgp = DeepGPSkorch(module=DeepGaussianProcess,
                       module__input_size=x_train.shape[-1],
                       module__hidden_size=[1],
                       module__output_size=y_train.shape[-1] * 2,
                       module__num_inducing=128,
                       lr=0.01,
                       max_epochs=1000,
                       batch_size=256,
                       train_split=None,
                       optimizer=torch.optim.Adam,
                       num_data=x_train.shape[0],
                       device=device)

    dgp.fit(x_train, y_train)

    pred = dgp.predict(x_test)
    pred_mean = pred[..., 0]
    pred_var = pred[..., 1]
    pred_epis_var = pred[..., 2]
    pred_aleo_var = pred[..., 3]

    plot_all(pred_mean, pred_var, np.sqrt(pred_epis_var), np.sqrt(pred_aleo_var))


def plot_all(pred_mean, pred_var, pred_epis_std, pred_aleo_std):
    # plot data
    ax = plt.subplot(2, 2, 1)
    ax.set_title('aleatoric')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, pred_aleo_std, dx, dy, ax)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('epistemic')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, pred_epis_std, dx, dy, ax)
    ax = plt.subplot(2, 1, 2)
    ax.set_title('combined')
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, np.sqrt(pred_var), dx, dy, ax)
    plt.show()


def plot_single(pred_mean, pred_std):
    ax = plt.subplot(1, 1, 1)
    plot(obs_aleo_x, obs_aleo_het_y, dx_test, pred_mean, pred_std, dx, dy, ax)
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


if __name__ == '__main__':
    main()
