# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pingouin import power_ttest as ptest
import math

import numpy as np
import pandas as pd
try:
    import statsmodels.stats.api as sms
except ImportError:
    print("Instal statsmodels using `pip install statsmodels==0.12.2")


class SignificanceTest(object):
    """This is the base class for the significance tests."""

    def __init__(self, **kwargs):
        pass

    def solve(self, **kwgs):
        """Interface of the solver for the significance test."""
        raise NotImplementedError("must be implemented in derived classes")


class ProportionSignificanceTest(SignificanceTest):
    """Signifiance test for 1 or 2 independent samples with proportion-format metrics.

    Using pooled variance for 2 sample tests.
    """

    def __init__(self, **kwds):
        super(ProportionSignificanceTest, self).__init__(**kwds)

    def check_params(self, **kwargs):
        """Check the parameters."""
        params = dict(
            sig_level=kwargs.get("sig_level", None),
            power=kwargs.get("power", None),
            p1=kwargs.get("p1", None),
            p2=kwargs.get("p2", None),
            n1=kwargs.get("n1", None),
            n2=kwargs.get("n2", None),
            alternative=kwargs.get("alternative", "two-sided"),
        )
        assert (
            params["sig_level"] is None
            or params["sig_level"] < 1
            and params["sig_level"] > 0
        ), "sig_level should be between (0, 1)"
        assert (
            params["power"] is None or params["power"] < 1 and params["power"] > 0
        ), "power should be between (0, 1)"
        assert (
            params["n1"] is None or params["n1"] > 0
        ), "n1 should be a positive number"
        assert (
            params["n2"] is None or params["n2"] > 0
        ), "n2 should be a positive number"
        assert (
            params["p1"] is None or params["p1"] < 1 and params["p1"] > 0
        ), "p1 should be between (0, 1)"
        assert (
            params["p2"] is None or params["p2"] < 1 and params["p2"] > 0
        ), "p2 should be between (0, 1)"
        assert params["alternative"] in (
            "two-sided",
            "larger",
            "smaller",
        ), "alternative should take value of '1' (1-sided) or '2' (2-sided)"

        if not params["n1"] and not params["n2"]:
            del params["n2"]
        num_nones = np.sum([params[k] is None for k in params if k != "alternative"])
        assert (
            num_nones == 1
        ), "1 and only 1 param of (n, p1, p2, power, sig_level) should be None"

    def inv_proportion_effectsize(self, effect_size, p1, p2):
        """compute p1 or p2 given a known effect size.

        Effect size is defined as:

        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

        ref: http://www.statmethods.net/stats/power.html
        """
        if p1:
            return np.sin(np.arcsin(np.sqrt(p1)) - effect_size / 2) ** 2
        else:
            return np.sin(np.arcsin(np.sqrt(p2)) + effect_size / 2) ** 2

    def _solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative="two-sided"):
        """wrapper of solver_power()"""
        try:
            val = sms.NormalIndPower().solve_power(
                effect_size=effect_size,
                nobs1=nobs1,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            return val
        except Exception as e:
            return 0

    def solve_2sample(self, p1=None, p2=None, n1=None, n2=None, sig_level=None, power=None,
                      alternative="two-sided", **kwargs):
        """solve for any one parameter of the power of a two sample proportion test.

        Parameters
        ----------
        p1 : float in interval (0,1), metric value for sample 1.
        p2 : float in interval (0,1), metric value for sample 2.
        n1 : int or float or None, number of observations of sample 1.
        n2 : int or float or None, number of observations of sample 2.
        sig_level : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I error or false positive ratio,
            which is the probability that the test fails to reject the Null Hypothesis if it is false.
        power : float in interval (0,1)
            power of the test, e.g. 0.85, is (1 - probability of a type II error)
            or equvilent to true positive ratio, which is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : string, "two-sided" (default), "larger", "smaller"
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either "larger", "smaller".

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.

        """
        self.check_params(
            p1=p1,
            p2=p2,
            n1=n1,
            n2=n2,
            sig_level=sig_level,
            power=power,
            alternative=alternative,
        )

        # estimate the minimum size for one of the two samples given other info
        if n1 and n2:
            ratio = n2 / n1
        elif n1 or n2:
            effect_size = sms.proportion_effectsize(p1, p2)
            ratio = self._solve_power(
                effect_size=effect_size,
                nobs1=n1 if n1 else n2,
                alpha=sig_level,
                power=power,
                ratio=None,
                alternative=alternative,
            )
            return (n1 if n1 else n2) * ratio, None
        else:
            effect_size = sms.proportion_effectsize(p1, p2)
            return self._solve_power(
                effect_size=effect_size,
                nobs1=None,
                alpha=sig_level,
                power=power,
                ratio=1.0,
                alternative=alternative,
            ), None

        # estimate the minimum proportion value for one of the two samples given other info.
        if p1 and p2:
            effect_size = sms.proportion_effectsize(p1, p2)
        else:
            effect_size = self._solve_power(
                nobs1=n1,
                alpha=sig_level,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            return self.inv_proportion_effectsize(effect_size, p1, p2), None
        # estimate the power or the significance level
        estimate = self._solve_power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=sig_level,
            power=power,
            ratio=ratio,
            alternative=alternative,
        )

        suggestions = []
        # suggest a new sample size if the estimated power or sig_level is not good.
        min_power = kwargs.get("min_power", 0.85)
        if not power and estimate < min_power * 0.999:
            new_n = self._solve_power(
                effect_size=effect_size,
                alpha=sig_level,
                power=min_power,
                ratio=ratio,
                alternative=alternative,
            )
            suggestion = {}
            suggestion["message"] = "The calculated power is {estimate:.4f} (< min_power {min_power}); " \
                "suggested new sample size n1 = n2 = {np.ceil(new_n)}"
            suggestion["n1"] = np.ceil(new_n)
            suggestion["n2"] = np.ceil(new_n)
            suggestions.append(suggestion)

        max_sig_level = kwargs.get("max_sig_level", 0.05)
        if not sig_level and estimate > max_sig_level * 1.001:
            new_n = self._solve_power(
                effect_size=effect_size,
                alpha=max_sig_level,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            suggestion = {}
            suggestion["message"] = f"The calculated sig_level is {estimate:.4f} (> max_sig_level {max_sig_level}); " \
                "suggested new sample size n1 = n2 = {np.ceil(new_n)}"
            suggestion["n1"] = np.ceil(new_n)
            suggestion["n2"] = np.ceil(new_n)
            suggestions.append(suggestion)

        return estimate, suggestions


class PowerBTest(SignificanceTest):
    """Paired ttest for paired samples whose values are in soft numeric format."""

    def __init__(self, **kwds):
        super(PowerBTest, self).__init__(**kwds)

    def calc_1pair_bt(self, scores1, scores2):
        """calculate the BT score for 1 pair of samples"""
        comp = (scores1 - scores2) > 0
        return np.sum(comp) / len(comp)

    def BT(self, df, epsilon=1e-6, max_iters=50) -> np.array:
        """calculate BT scores for multi-pairs of samples iteratively

        ref: https://en.wikipedia.org/wiki/Bradley-Terry_model
        """

        def _BT_pair(array_0, array_1):
            return self.calc_1pair_bt(array_0, array_1), self.calc_1pair_bt(array_1, array_0)

        def _safe_divide(x, y):
            return 0 if np.isclose(y, 0) else x/y

        assert isinstance(df, pd.DataFrame) and df.shape[1] > 1, "df should be a pd.DataFrame with more than 1 columns"

        if df.shape[1] == 2:
            return np.array(_BT_pair(df.iloc[:, 0], df.iloc[:, 1]))

        # calculate initial BT scores
        bt = [[0.5] * df.shape[1] for i in range(df.shape[1])]
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i <= j:
                    continue
                bt[i][j], bt[j][i] = _BT_pair(df.iloc[:, i], df.iloc[:, j])

        # iterate to converge
        p, p0 = np.array([1.0 / df.shape[1]] * df.shape[1]), np.array([0] * df.shape[1])
        for _ in range(max_iters):
            for i in range(df.shape[1]):
                numerator = np.sum([v for k, v in enumerate(bt[i]) if k != i])
                denorm = np.sum([
                    _safe_divide((bt[i][k] + bt[k][i]), (p[i] + p[k]))
                    for k in range(df.shape[1]) if k != i
                ])
                p[i] = _safe_divide(numerator, denorm)
            p = p / np.sum(p)

            if np.max(np.abs(p - p0)) < epsilon:
                return p
            p0 = p
        return p

    def solve_2sample(self, scores1, scores2, sig_level=None, power=None,
                      alternative="two-sided", **kwargs):
        """solve for any one parameter of the power function of a two sample proportion test.

        Parameters
        ----------
        scores1 : list of float values, scores of sample 1.
        scores2 : list of float values, scores of sample 2.
        sig_level : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I error or false positive ratio,
            which is the probability that the test fails to reject the Null Hypothesis if it is false.
        power : float in interval (0,1)
            power of the test, e.g. 0.85, is (1 - probability of a type II error)
            or equvilent to true positive ratio, which is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : string, "two-sided" (default), "larger", "smaller"
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either "larger", "smaller".

        Note
        ----
        Exactly ONE of the parameters power and sig_level must be passed as None.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.
        """
        assert (len(scores1) == len(scores2)), "length of paired samples should be equal."

        p = self.calc_1pair_bt(scores1, scores2)
        try:
            return ProportionSignificanceTest().solve_2sample(
                n1=len(scores1)-1, n2=len(scores2)-1, p1=p, p2=0.5, power=power,
                sig_level=sig_level, alternative=alternative,
            )
        except Exception as e:
            return 0, [str(e)]


class PowerTTest(SignificanceTest):
    """Paired ttest for paired samples whose values are in soft numeric format."""

    def __init__(self, **kwds):
        super(PowerTTest, self).__init__(**kwds)

    def solve(self, d=None, std=1, n=None, sig_level=None, power=None,
              alternative="two-sided", **kwargs):
        """solve for any one parameter of the power function in an one sample proportion test.

        ref: https://pingouin-stats.org/generated/pingouin.power_ttest.html

        Parameters
        ----------
        d : float, effect size of the paired test.
        std : float, std of effect size, default=1.
        n : int or float or None, number of observations of paired samples.
        sig_level : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I error or false positive ratio,
            which is the probability that the test fails to reject the Null Hypothesis if it is false.
        power : float in interval (0,1)
            power of the test, e.g. 0.85, is (1 - probability of a type II error)
            or equvilent to true positive ratio, which is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : string, "two-sided" (default), "larger", "smaller"
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either "larger", "smaller".

        Note
        ----
        Exactly ONE of the parameters d, n, power and sig_level must be passed as None,
        and that parameter is determined from the others.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.
        """
        effect_size = d / std if d and std > 0 else 0
        try:
            return ptest(
                d=effect_size, n=n, power=power, alpha=sig_level, alternative=alternative
            ), []
        except Exception as e:
            return 0, [str(e)]

    def solve_2sample(self, scores1, scores2, sig_level=None, power=None,
                      alternative="two-sided", **kwargs):
        """solve for any one parameter of the power function of a two sample proportion test.
        """
        assert (len(scores1) == len(scores2)), "length of paired samples should be equal."

        scores1, scores2 = np.array(scores1), np.array(scores2)
        d = np.mean(scores1 - scores2)
        std = np.std(scores1 - scores2)
        return self.solve(
            d=d, std=std, n=len(scores1), sig_level=sig_level, power=power, alternative=alternative
        )


# shortcut functions
power_p_test_2sample = ProportionSignificanceTest().solve_2sample
power_t_test = PowerTTest().solve
power_t_test_2sample = PowerTTest().solve_2sample
power_bt_test_2sample = PowerBTest().solve_2sample
BT = PowerBTest().BT