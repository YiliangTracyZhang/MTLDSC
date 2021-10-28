#!/usr/bin/python
from __future__ import division
import collections
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import linear_model
from numpy.linalg import inv

def calculate(gwas_snps, ld_scores, N1, intercept):
    ld_scores = ld_scores.drop(['CHR', 'BP', 'CM', 'MAF'], axis=1, errors='ignore').reset_index(drop=True)
    merged = pd.merge(gwas_snps, ld_scores, on=['SNP'])

    ld_score_all = merged.iloc[:,4]
    Z_x, Z_y = merged['Z_x'], merged['Z_y']
    p0 = len(merged)

    h1_2 = p0 * (np.mean(Z_x ** 2) - 1) / (N1 * np.mean(ld_score_all))
    aarho = p0 * (np.mean(Z_y ** 2) - intercept) / np.mean(ld_score_all)
    arho = np.sum(Z_x * Z_y) / (np.sqrt(N1) * np.mean(ld_score_all))

    w1 = 1 + N1 * h1_2 * ld_score_all / p0
    w2 = intercept + aarho * ld_score_all / p0

    wh1 = 1 / (w1 ** 2)
    wh2 = 1 / (w2 ** 2)

    h1_2 = (np.sum(ld_score_all * wh1 * (Z_x ** 2 - 1)) / np.sum((ld_score_all ** 2) * wh1)) * (p0 / N1)
    aarho_m = linear_model.LinearRegression().fit(pd.DataFrame(ld_score_all), pd.DataFrame(Z_y ** 2), sample_weight=wh2)

    intercept = aarho_m.intercept_[0]
    aarho = aarho_m.coef_[0][0] * p0

    w1 = 1 + N1 * h1_2 * ld_score_all / p0
    w2 = intercept + aarho * ld_score_all / p0
    w3 = np.sqrt(N1) * ld_score_all * arho / p0
    wh1 = 1 / (w1 ** 2)
    wh2 = 1 / (w2 ** 2)
    w = 1 / (w1 * w2 + w3 * w3)

    m = linear_model.LinearRegression().fit(pd.DataFrame(ld_score_all), pd.DataFrame(Z_x * Z_y), sample_weight=w)
    corr_pheno = m.intercept_[0]
    arho = m.coef_[0][0] * p0 / np.sqrt(N1)

    w3 = np.sqrt(N1) * ld_score_all * arho / p0 + corr_pheno
    w = 1 / (w1 * w2 + w3 * w3)

    # Calculate Jackknife variance estimate
    nblock = 200
    cov_block = np.empty(nblock)
    # h1_block = np.empty(nblock)
    # h2_block = np.empty(nblock)
    
    # h1_2x_tot = np.sum((ld_score_all ** 2) * wh1)
    # h1_2y_tot = np.sum(ld_score_all * wh1 * (Z_x ** 2 - 1))
    # Xwh = np.vstack([wh2, wh2 * ld_score_all])
    # h2_2x_tot = Xwh.dot(np.vstack([np.ones(len(ld_score_all)), ld_score_all]).T)
    # h2_2y_tot = Xwh.dot(Z_y ** 2)
    Xw = np.vstack([w, w * ld_score_all])
    cov_x_tot = Xw.dot(np.vstack([np.ones(len(ld_score_all)), ld_score_all]).T)
    cov_y_tot = Xw.dot(Z_x * Z_y)

    # for j, (ldscore_b, Z_x_b, Z_y_b, wh1_b, wh2_b, w_b) in enumerate(zip(np.array_split(ld_score_all, nblock),
    #     np.array_split(Z_x, nblock), np.array_split(Z_y, nblock), np.array_split(wh1, nblock),
    #     np.array_split(wh2, nblock), np.array_split(w, nblock))):
    for j, (ldscore_b, Z_x_b, Z_y_b, w_b) in enumerate(zip(np.array_split(ld_score_all, nblock),
        np.array_split(Z_x, nblock), np.array_split(Z_y, nblock), np.array_split(w, nblock))):
        # h1_2x_curr = h1_2x_tot - np.sum((ldscore_b ** 2) * wh1_b)
        # h1_2y_curr = h1_2y_tot - np.sum(ldscore_b * wh1_b * (Z_x_b ** 2 - 1))
        # Xwh_curr = np.vstack([wh2_b, wh2_b * ldscore_b])
        # h2_2x_curr = h2_2x_tot - Xwh_curr.dot(np.vstack([np.ones(len(ldscore_b)), ldscore_b]).T)
        # h2_2y_curr = h2_2y_tot - Xwh_curr.dot(Z_y_b ** 2)
        Xw_curr = np.vstack([w_b, w_b * ldscore_b])
        cov_x_curr = cov_x_tot - Xw_curr.dot(np.vstack([np.ones(len(ldscore_b)), ldscore_b]).T)
        cov_y_curr = cov_y_tot - Xw_curr.dot(Z_x_b * Z_y_b)

        # h1_2_curr = (h1_2y_curr * p0 / h1_2x_curr) / N1
        # h2_2_curr = inv(h2_2x_curr).dot(h2_2y_curr)[1] * p0
        cov_curr = inv(cov_x_curr).dot(cov_y_curr)[1] * p0 / np.sqrt(N1)

        cov_block[j] = cov_curr
        # h1_block[j] = h1_2_curr
        # h2_block[j] = h2_2_curr
    
    # genetic correlation
    corr = arho / np.sqrt(aarho * h1_2)

    if np.isnan(corr).any():
        print('Some correlation estimates are NaN because the heritability '
              'estimates were negative.')

    # p-value and standard error
    se_cov = np.sqrt(np.var(cov_block) * (nblock - 1))
    se_corr = se_cov / np.sqrt(aarho * h1_2)
    p_value = norm.sf(abs(corr / se_corr)) * 2

    out = pd.DataFrame(collections.OrderedDict(
        [('corr', [corr]),
         ('se', [se_corr]),
         ('pvalue', [p_value]),
         ('p0', [p0])
        ]
    ))

    return out

