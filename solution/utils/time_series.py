import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm


def time_series_info(data: pd.Series) -> pd.DataFrame:
    """
    Оценивает ряд на стационарность, гетероскопичность, автокореляцию, нормальность.
    """
    mean_pval = np.NAN
    stat_adfuller_pval = np.NAN
    stat_kpss_pval = np.NAN
    norm_pval = np.NAN
    stat_sign_autocor_cnt = np.NAN
    ff_pval = np.NAN

    try:
        _, mean_pval = ttest_1samp(data, 0)
    except:
        print("Failed calculate mean_pval")
    try:
        _, stat_adfuller_pval = adfuller(data)[:2]
    except:
        print("Failed calculate stat_adfuller_pval")
    try:
        _, stat_kpss_pval = kpss(data, regression="c", nlags="auto")[:2]
    except:
        print("Failed calculate stat_kpss_pval")
    try:
        _, norm_pval = shapiro(data)
    except:
        print("Failed calculate norm_pval")
    try:
        stat_sign_autocor_cnt = (acorr_ljungbox(data, model_df=0, return_df=True)['lb_pvalue'] < 0.05).sum()
    except:
        print("Failed calculate stat_sign_autocor_cnt")
    try:
        _, _, _, ff_pval = het_breuschpagan(data, sm.add_constant(data.reset_index().index))
    except:
        print("Failed calculate ff_pval")

    no_stat = stat_adfuller_pval <= 0.05
    is_stat = stat_kpss_pval > 0.05
    is_no_autocorr = stat_sign_autocor_cnt == 0
    is_homosced = ff_pval > 0.05
    is_mean_zero = mean_pval > 0.05
    is_norm = norm_pval > 0.05

    total = no_stat and is_stat and is_no_autocorr and is_homosced and is_mean_zero and is_norm

    df_data = {
        "title": ["no_stationary", "is_stationary", "no_autocorrelation", "is_homoscedasticity", "is_mean_zero",
                  "is_normality", "total"],
        "method": ["adfuller", "kpss", "acorr_ljungbox", "het_breuschpagan", "ttest_1samp", "shapiro", "-"],
        "val": [stat_adfuller_pval, stat_kpss_pval, stat_sign_autocor_cnt, ff_pval, mean_pval, norm_pval, np.NAN],
        "check": ["<=0.05", ">0.05", "==0", ">0.05", ">0.05", ">0.05", "-"],
        "result": [no_stat, is_stat, is_no_autocorr, is_homosced, is_mean_zero, is_norm, total],
        "opinion": [
            "✅" if no_stat else "❌",
            "✅" if is_stat else "❌",
            "✅" if is_no_autocorr else "❌",
            "✅" if is_homosced else "❌",
            "✅" if is_mean_zero else "❌",
            "✅" if is_norm else "❌",
            "✅" if total else "❌",
        ],
    }
    result = pd.DataFrame(df_data)
    return result
