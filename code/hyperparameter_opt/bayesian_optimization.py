from skopt import BayesSearchCV
import numpy as np
from scipy.stats import multivariate_normal

counter = 0
opt = None


def bayesian_optimization(model, space, scorer, x_train, y_train, x_test, y_test, n_iter=256, cv=3, n_jobs=None):
    global counter
    global opt

    if n_jobs is None:
        n_jobs = cv

    opt = BayesSearchCV(
        model,
        space,
        scoring=scorer,
        n_iter=n_iter,
        cv=cv,
        verbose=10,
        n_jobs=n_jobs)

    counter = 0
    opt.fit(x_train, y_train, callback=on_step)

    print(opt.best_params_)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))


# callback handler
def on_step(optim_result):
    global counter
    counter = counter + 1

    score = opt.best_score_
    print(opt.best_params_)
    print("best score: %s" % score)
    print('counter: %d/%d' % (counter, opt.n_iter))


def mse_scorer(estimator, x, y):
    y_predicted = estimator.predict(x)
    if len(y_predicted.shape) != len(y.shape):
        y_predicted = y_predicted[..., 0]
    return -np.sqrt(np.mean((y - y_predicted) ** 2))


def crps_scorer(estimator, x, y):
    # Adapted from Equation 5:
    #     Calibrated Probablistic Forecasting Using Ensemble Model Output
    #     Statistics and Minimum CRPS Estimation
    #     http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1

    y_predicted = estimator.predict(x)

    mu = y_predicted[..., 0]
    sigma = np.sqrt(y_predicted[..., 1])

    sx = (y - mu) / sigma

    normal = multivariate_normal(0, 1)
    pdf = normal.pdf(sx)
    cdf = normal.cdf(sx)

    crps = sigma * (sx * (2 * cdf - 1) + 2 * pdf - (1 / np.sqrt(np.pi)))

    return -np.mean(crps)
