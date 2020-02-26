import warnings

from numpy.linalg import pinv
from statsmodels.regression.quantile_regression import QuantReg, QuantRegResults
from statsmodels.regression.linear_model import RegressionResultsWrapper
import numpy as np
from statsmodels.regression.quantile_regression import kernels, hall_sheather, bofinger, chamberlain
import scipy.stats as stats
from scipy.stats import norm


# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
#
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
#
# Copyright (c) 2009-2018 statsmodels Developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of statsmodels nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

'''
Quantile regression model

Model parameters are estimated using iterated reweighted least squares. The
asymptotic covariance matrix estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
'''

# Override statsmodels Quantile Regression for a simple fix to work with
# singular matrix, needed to copy code from statsmodels to do this, see
# copyright notice above!
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning


class QuantRegFix(QuantReg):

    def fit(self, q=.5, vcov='robust', kernel='epa', bandwidth='hsheather',
            max_iter=1000, p_tol=1e-6, **kwargs):
        """
        Solve by Iterative Weighted Least Squares

        Parameters
        ----------
        q : float
            Quantile must be between 0 and 1
        vcov : str, method used to calculate the variance-covariance matrix
            of the parameters. Default is ``robust``:

            - robust : heteroskedasticity robust standard errors (as suggested
              in Greene 6th edition)
            - iid : iid errors (as in Stata 12)

        kernel : str, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth : str, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        """

        if q < 0 or q > 1:
            raise Exception('p must be between 0 and 1')

        kern_names = ['biw', 'cos', 'epa', 'gau', 'par']
        if kernel not in kern_names:
            raise Exception("kernel must be one of " + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]

        if bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        elif bandwidth == 'chamberlain':
            bandwidth = chamberlain
        else:
            raise Exception("bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'")

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        # use dimensionality instead of rank to work with singular matrix
        exog_rank = self.exog.shape[-1]
        self.rank = exog_rank
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = self.nobs - self.rank
        n_iter = 0
        xstar = exog

        beta = np.ones(exog_rank)
        # TODO: better start, initial beta is used only for convergence check

        # Note the following does not work yet,
        # the iteration loop always starts with OLS as initial beta
        # if start_params is not None:
        #    if len(start_params) != rank:
        #       raise ValueError('start_params has wrong length')
        #       beta = start_params
        #    else:
        #       # start with OLS
        #       beta = np.dot(np.linalg.pinv(exog), endog)

        diff = 10
        cycle = False

        history = dict(params=[], mse=[])
        while n_iter < max_iter and diff > p_tol and not cycle:
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)

            mask = np.abs(resid) < .000001
            resid[mask] = ((resid[mask] >= 0) * 2 - 1) * .000001
            resid = np.where(resid < 0, q * resid, (1 - q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))
            history['params'].append(beta)
            history['mse'].append(np.mean(resid * resid))

            if (n_iter >= 300) and (n_iter % 100 == 0):
                # check for convergence circle, should not happen
                for ii in range(2, 10):
                    if np.all(beta == history['params'][-ii]):
                        cycle = True
                        warnings.warn("Convergence cycle detected", ConvergenceWarning)
                        break

        if n_iter == max_iter:
            warnings.warn("Maximum number of iterations (" + str(max_iter) +
                          ") reached.", IterationLimitWarning)

        e = endog - np.dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        # h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog),
                iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1. / (nobs * h) * np.sum(kernel(e / h))

        if vcov == 'robust':
            d = np.where(e > 0, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = xtxi @ xtdx @ xtxi
        elif vcov == 'iid':
            vcov = (1. / fhat0) ** 2 * q * (1 - q) * pinv(np.dot(exog.T, exog))
        else:
            raise Exception("vcov must be 'robust' or 'iid'")

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)

        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1. / fhat0
        lfit.bandwidth = h
        lfit.history = history

        return RegressionResultsWrapper(lfit)