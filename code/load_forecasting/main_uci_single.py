import torch

from evaluation.evaluate_forecasting_util import evaluate_single
from load_forecasting.forecast_util import dataset_df_to_np
from load_forecasting.post_processing import recalibrate
from load_forecasting.predict import predict_transform
from models.deep_ensemble_sklearn import DeepEnsemble
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from training.training_util import load_train
from util.data.data_src_tools import load_uci_load
from util.data.data_tools import preprocess_load_data_forec
import workalendar.europe as wk

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    # load and pre process data
    load_df = load_uci_load()

    calendars = [wk.Portugal()]
    train_df, test_df, scaler = preprocess_load_data_forec(load_df, short_term=False, quarter_hour=True, n_ahead=1, calendars=calendars)

    x_train, y_train, offset_train = dataset_df_to_np(train_df)
    x_test, y_test, offset_test = dataset_df_to_np(test_df)

    y_test_orig = scaler.inverse_transform(y_test) + offset_test
    y_train_orig = scaler.inverse_transform(y_train) + offset_train

    hs = [132, 77, 50]
    lr = 5.026e-05
    epochs = 1253
    # epochs = 100

    # initialize model
    ensemble_model = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        hidden_size=hs,
        lr=lr,
        max_epochs=epochs,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device
    )

    # train and recalibrate
    load_train(ensemble_model, x_train, y_train, 'deep_ens', '../trained_models/', 'uci_load_forecasting_', False)

    pred_mean_train, pred_var_train, _, _, _ = predict_transform(ensemble_model, x_train, scaler, offset_train, 'Deep Ensemble UCI')
    recal = recalibrate(pred_mean_train, pred_var_train, y_train_orig)

    # predict
    pred_mean, pred_var, _, _, _ = predict_transform(ensemble_model, x_test, scaler, offset_test, 'Deep Ensemble UCI')
    pred_mean, pred_var = recal(pred_mean, pred_var)

    # evaluate
    evaluate_single(pred_mean, pred_var, y_test_orig)


if __name__ == '__main__':
    main()
