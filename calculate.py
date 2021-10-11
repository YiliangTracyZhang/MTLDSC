#!/usr/bin/python
from __future__ import division
import collections
from itertools import product
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import norm
from numpy.linalg import inv

def calculate(gwas_snps, ld_scores, annots, N1, N2):
    np.seterr(invalid='ignore')

    ###   Clean up data   ###
    if annots is None:
        annot = ld_scores[['SNP']]
        annot['ALL_'] = 1
    else:
        annot = pd.concat(annots)

    ld_snps = set(ld_scores['SNP'])
    annot = annot.loc[annot['SNP'].isin(ld_snps)].reset_index(drop=True)

    ld_scores = ld_scores.drop(['CHR', 'BP', 'CM', 'MAF'], axis=1, errors='ignore').reset_index(drop=True)
    annot = annot.drop(['BP', 'SNP', 'CHR', 'CM'], axis=1, errors='ignore')
    gwas_snps.drop(['idx'], axis=1, errors='ignore', inplace=True)

    num_annotations = len(annot.columns)

    merged = pd.merge(gwas_snps,
                      pd.concat([ld_scores, annot], axis=1),
                      on=['SNP'])

    ld_score_all = merged.iloc[:,4]
    if num_annotations == 1:  # non-stratified analysis
        ld_scores = merged.iloc[:,4:5]
        annot = merged.iloc[:,5:6]

    else:  # we added in an all 1's column in prep step, so exclude that
        ld_scores = merged.iloc[:,5:4 + num_annotations]
        annot = merged.iloc[:,5 + num_annotations: 4 + 2 * num_annotations]
        num_annotations -= 1

    ###   Calculate genetic correlation   ###
    # Calculate S and W matrix
    P = annot.sum()
    p0 = len(ld_scores)

    S = np.empty([num_annotations, num_annotations])
    for i, j in product(range(num_annotations), range(num_annotations)):
        S[i][j] = np.sum(ld_scores[annot.iloc[:,i] == 1].iloc[:,j]) / (P[i] * P[j])

    W = np.empty([num_annotations, num_annotations])
    for i, j in product(range(num_annotations), range(num_annotations)):
        W[i][j] = np.sum((annot.iloc[:,i]==1) & (annot.iloc[:,j]==1)) / np.sum(annot.iloc[:,j] == 1)

    # Calculate heritability
    Z_x, Z_y = merged['Z_x'], merged['Z_y']

    h2_1 = np.array([p0 * (np.mean(Z_x ** 2) - 1) / (N1 * np.mean(ld_score_all))])
    h2_2 = np.array([p0 * (np.mean(Z_y ** 2) - 1) / (N2 * np.mean(ld_score_all))])

    # Calculate sample overlap correction
    w1 = 1 + N1 * (h2_1 * ld_score_all / len(ld_score_all))
    w2 = 1 + N2 * (h2_2 * ld_score_all / len(ld_score_all))

    w3 = np.mean(Z_x * Z_y) * ld_score_all
    w = 1 / (w1 * w2 + w3 * w3)

    # Calculate Jackknife variance estimate
    nblock = 200
    q_block = np.empty([num_annotations, nblock])
    h2_1_block = np.empty([num_annotations, nblock])
    h2_2_block = np.empty([num_annotations, nblock])

    m = linear_model.LinearRegression().fit(pd.DataFrame(ld_score_all), pd.DataFrame(Z_x * Z_y), sample_weight=w)
    corr_pheno = m.intercept_[0]

    for i in range(num_annotations):
        df_x = Z_x[annot.iloc[:,i] == 1]
        df_y = Z_y[annot.iloc[:,i] == 1]
        tot = np.dot(df_x, df_y)
        tot1 = np.dot(df_x, df_x)
        tot2 = np.dot(df_y, df_y)
        for j, (b_x, b_y) in enumerate(zip(np.array_split(df_x, nblock), np.array_split(df_y, nblock))):
            q_block[i][j] = (tot - np.dot(b_x, b_y)) / ((len(df_x) - len(b_x)) * ((N1 * N2) ** 0.5))
            h2_1_block[i][j] = (tot1 - np.dot(b_x, b_x)) / ((len(df_x) - len(b_x)) * N1) - 1 / N1
            h2_2_block[i][j] = (tot2 - np.dot(b_y, b_y)) / ((len(df_y) - len(b_y)) * N2) - 1 / N2

    h2_1 = np.mean(h2_1_block, axis=1)
    h2_2 = np.mean(h2_2_block, axis=1)

    # rho
    rho_block = W.dot(inv(S)).dot(q_block)
    rho_corrected = W.dot(inv(S)).dot(q_block - corr_pheno / ((N1 * N2) ** 0.5))
    h2_1_block = W.dot(inv(S)).dot(h2_1_block)
    h2_2_block = W.dot(inv(S)).dot(h2_2_block)
    corr_block = rho_block / (h2_1_block * h2_2_block) ** 0.5
    corr_adjust = rho_corrected / (h2_1_block * h2_2_block) ** 0.5

    # covariance of rho
    
    rho = np.mean(rho_block, axis=1)
    rho_corrected = np.mean(rho_corrected, axis=1)
    
    # genetic correlation
    corr = np.mean(corr_block)
    corr_corrected = np.mean(corr_adjust)

    if np.isnan(corr).any() or np.isnan(corr_corrected).any():
        print('Some correlation estimates are NaN because the heritability '
              'estimates were negative.')

    # p-value and standard error
    cov_rho = np.array([np.cov(rho_block, bias=True) * (nblock - 1)], ndmin=2)
    se_rho = cov_rho.diagonal() ** 0.5
    cov_corr = np.array([np.cov(corr_block, bias=True) * (nblock - 1)], ndmin=2)
    se_corr = cov_corr.diagonal() ** 0.5
    cov_corr_adjust = np.array([np.cov(corr_adjust, bias=True) * (nblock - 1)], ndmin=2)
    se_corr_adjust = cov_corr_adjust.diagonal() ** 0.5
    p_value = norm.sf(abs(rho / se_rho)) * 2
    p_value_corrected = norm.sf(abs(rho_corrected / se_rho)) * 2
    p_value_corr = norm.sf(abs(corr / se_corr)) * 2
    p_value_corrected_corr = norm.sf(abs(corr_corrected / se_corr_adjust)) * 2

    out = pd.DataFrame(collections.OrderedDict(
        [('rho', rho),
         ('rho_corrected', rho_corrected),
         ('se_rho', se_rho),
         ('pvalue_cov', p_value),
         ('pvalue_corrected_cov', p_value_corrected),
         ('corr', corr),
         ('se_corr', se_corr),
         ('corr_corrected', corr_corrected),
         ('se_corr_corrected', se_corr_adjust),
         ('pvalue_corr', p_value_corr),
         ('pvalue_corrected_corr', p_value_corrected_corr),
         ('h2_1', h2_1),
         ('h2_2', h2_2),
         ('p', P),
         ('p0', p0)
        ]
    ))
    # Check for all-1 annotations and remove them from the output
    has_all_ones = False
    if len(out) > 1:
        for row in out.index:
            if annot[row].all():
                out.loc[row,:-2] = np.nan
                has_all_ones = True
    if has_all_ones:
        print('NOTE: There is at least one annotation that applies to every SNP. '
              'Non-stratified analysis will provide better estimates for the '
              'total genetic covariance and genetic correlation, so we have labeled '
              'the results for these annotations as "NA" in the output.')

    return out

