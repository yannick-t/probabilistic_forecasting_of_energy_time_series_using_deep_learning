import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.calibration import probabilistic_calibration_multiple, marginal_calibration_multiple, interval_coverage, \
    probabilistic_calibration, marginal_calibration
from evaluation.scoring import rmse, mape, crps, log_likelihood
from evaluation.sharpness import sharpness_plot_multiple, sharpness_plot_histogram_joint_multiple, sharpness_avg_width, \
    sharpness_plot, sharpness_plot_histogram_joint
from util.model_enum import ModelEnum

# pretty names for plot titles etc.
names_pretty_dict = {ModelEnum.simple_nn_aleo.name: 'Simple NN', ModelEnum.concrete.name: 'Concrete',
                     ModelEnum.fnp.name: 'FNP',
                     ModelEnum.deep_ens.name: 'Deep Ens.', ModelEnum.bnn.name: 'BNN', ModelEnum.dgp.name: 'Deep GP',
                     ModelEnum.linear_reg.name: 'Linear Regression',
                     ModelEnum.quantile_reg.name: 'Quantile Regression'}


def evaluate_multiple(names, pred_means, pred_vars, true_y, pred_ood_vars, result_folder, result_prefix):
    names_pretty = [names_pretty_dict[name] for name in names]

    # calibration
    probabilistic_calibration_multiple(names_pretty, pred_means, pred_vars, true_y)
    plt.savefig(result_folder + result_prefix + 'calibration_probabilistic.pdf')
    marginal_calibration_multiple(names_pretty, pred_means, pred_vars, true_y)
    plt.savefig(result_folder + result_prefix + 'calibration_marginal.pdf')

    # sharpness
    sharpness_plot_multiple(names_pretty, pred_vars)
    plt.savefig(result_folder + result_prefix + 'calibration_sharpness.pdf')

    # epistemic out of distribution evaluation
    for counter, p_ood in enumerate(pred_ood_vars):
        sharpness_plot_histogram_joint_multiple(names_pretty, pred_vars, p_ood)
        plt.savefig(result_folder + result_prefix + 'sharpness_ood' + str(counter) + '.pdf')

    plt.show()

    scores = pd.DataFrame(index=names, columns=['90IntCov', '50IntCov', 'AvgCent50W', 'AvgCent90W', 'RMSE', 'MAPE',
                                                'CRPS', 'AVGNLL'])

    # scoring etc.
    for name, pmean, pvar in zip(names, pred_means, pred_vars):
        print('#########################################')
        print('Model: ' + name)
        cov = interval_coverage(pmean, pvar, true_y, 0.9)
        print('0.9 interval coverage: %.5f' % cov)
        scores.loc[name, '90IntCov'] = cov

        cov = interval_coverage(pmean, pvar, true_y, 0.5)
        print('0.5 interval coverage: %.5f' % cov)
        scores.loc[name, '50IntCov'] = cov

        avg_5, avg_9 = sharpness_avg_width(pvar)
        print('Average central 50%% interval width: %.5f' % avg_5)
        print('Average central 90%% interval width: %.5f' % avg_9)
        scores.loc[name, 'AvgCent50W'] = avg_5
        scores.loc[name, 'AvgCent90W'] = avg_9

        scores.loc[name, 'RMSE'] = rmse(pmean, true_y).squeeze()
        scores.loc[name, 'MAPE'] = mape(pmean, true_y).squeeze()
        scores.loc[name, 'CRPS'] = crps(pmean, np.sqrt(pvar), true_y).squeeze()
        scores.loc[name, 'AVGNLL'] = -log_likelihood(pmean, np.sqrt(pvar), true_y).squeeze()

        print("RMSE: %.4f" % scores.loc[name, 'RMSE'])
        print("MAPE: %.2f" % scores.loc[name, 'MAPE'])
        print("CRPS: %.4f" % scores.loc[name, 'CRPS'])
        print("Average Negative LL: %.4f" % scores.loc[name, 'AVGNLL'])

    return scores


def evaluate_single(pred_mean, pred_var, true_y, pred_ood_var):
    # calibration
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Probabilistic Calibration: Probability Integral Transform Histogram')
    probabilistic_calibration(pred_mean, pred_var, true_y, ax)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Marginal Calibration: Difference between empirical CDF and average predictive CDF')
    marginal_calibration(pred_mean, pred_var, true_y, ax)

    # interval coverage
    cov = interval_coverage(pred_mean, pred_var, true_y, 0.9)
    print('0.9 interval coverage: %.5f' % cov)

    cov = interval_coverage(pred_mean, pred_var, true_y, 0.5)
    print('0.5 interval coverage: %.5f' % cov)

    # sharpness
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Sharpness: Predictive Interval Width Boxplot')
    sharpness_plot(pred_var, ax)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Sharpness: Predictive Interval Width Histogram')
    sharpness_plot_histogram_joint(pred_var, pred_ood_var, ax)
    avg_5, avg_9 = sharpness_avg_width(pred_var)
    print('Average central 50%% interval width: %.5f' % avg_5)
    print('Average central 90%% interval width: %.5f' % avg_9)

    # scoring
    print("RMSE: %.4f" % rmse(pred_mean, true_y))
    print("MAPE: %.2f" % mape(pred_mean, true_y))
    print("CRPS: %.4f" % crps(pred_mean, np.sqrt(pred_var), true_y))
    print("Average LL: %.4f" % log_likelihood(pred_mean, np.sqrt(pred_var), true_y))

    plt.show()


def evaluate_multi_step(pred_df, y_test, offset_test, scaler):
    rmses = np.empty(shape=(len(pred_df.columns)))
    mapes = np.empty(shape=(len(pred_df.columns)))
    for idx, column in enumerate(pred_df.columns):
        first_valid = pred_df.loc[:, column].first_valid_index()
        last_valid = pred_df.loc[:, column].last_valid_index()
        first_valid_idx = pred_df.index.get_loc(first_valid)
        last_valid_idx = pred_df.index.get_loc(last_valid) + 1

        pred = pred_df.loc[first_valid: last_valid, column].to_numpy().reshape(-1, 1)
        pred = scaler.inverse_transform(pred) + offset_test[first_valid_idx: last_valid_idx]
        y_test_adj = y_test[first_valid_idx: last_valid_idx]
        rmses[idx] = rmse(pred, y_test_adj)
        mapes[idx] = mape(pred, y_test_adj)

    print('############################')
    print('Multi-Step:')
    print("RMSE: %.4f" % rmses.mean())
    print("MAPE: %.2f" % mapes.mean())


def plot_test_data(pred_mean, pred_var, y_true, timestamp, ax):
    x = timestamp

    ax.plot(x, y_true.squeeze(), color='blue')
    ax.plot(x, pred_mean.squeeze(), color='orange')

    ax.plot(x, (y_true.squeeze() - pred_mean.squeeze()) ** 2, color='red')

    if pred_var is not None:
        std_deviations = np.sqrt(pred_var.squeeze())

        for j in range(1, 5):
            ax.fill_between(x, pred_mean.squeeze() - j / 2 * std_deviations,
                            pred_mean.squeeze() + j / 2 * std_deviations,
                            alpha=0.1, color='orange')
