import os

import numpy as np
import pandas as pd

def allign_alleles(df, Nsumstats):
    """Look for reversed alleles and inverts the z-score for one of them.

    Here, we take advantage of numpy's vectorized functions for performance.
    """
    d = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ref_allele = []
    for colname in ['A1_ref', 'A2_ref']:
        tmp = np.empty(len(df[colname]), dtype=int)
        for k, v in d.items():
            tmp[df[colname] == k] = v
        ref_allele.append(tmp)

    matched_or_reversed(df, ref_allele, 'x', d)
    for i in range(Nsumstats):
        matched_or_reversed(df, ref_allele, i, d)

def matched_or_reversed(df, ref_allele, suffix, d):
    allele = []
    A1_colname = "A1_{}".format(suffix)
    A2_colname = "A2_{}".format(suffix)
    for colname in [A1_colname, A2_colname]:
        tmp = np.empty(len(df[colname]), dtype=int)
        for k, v in d.items():
            tmp[df[colname] == k] = v
        allele.append(tmp)
    matched_alleles = (((ref_allele[0] == allele[0]) & (ref_allele[1] == allele[1])) | 
        ((ref_allele[0] == 3 - allele[0]) & (ref_allele[1] == 3 - allele[1])))
    reversed_alleles = (((ref_allele[0] == allele[1]) & (ref_allele[1] == allele[0])) |
        ((ref_allele[0] == 3 - allele[1]) & (ref_allele[1] == 3 - allele[0])))
    df.loc[:, "Z_{}".format(suffix)] *= -2 * reversed_alleles + 1
    df = df.loc[(matched_alleles|reversed_alleles)]
    ref_allele[0] = ref_allele[0][matched_alleles|reversed_alleles]
    ref_allele[1] = ref_allele[1][(matched_alleles|reversed_alleles)]

def get_files(file_name):
    if '@' in file_name:
        valid_files = []
        for i in range(1, 23):
            cur_file = file_name.replace('@', str(i))
            if os.path.isfile(cur_file):
                valid_files.append(cur_file)
            else:
                raise ValueError('No file matching {} for chr {}'.format(
                    file_name, i))
        return valid_files
    else:
        if os.path.isfile(file_name):
            return [file_name]
        else:
            ValueError('No files matching {}'.format(file_name))


def prep(bfile, sumstats1, sumstatslst, N1, Nlst):
    bim_files = get_files(bfile + '.bim')

    # read in bim files
    bims = [pd.read_csv(f,
                        header=None,
                        names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'],
                        delim_whitespace=True) for f in bim_files]
    bim = pd.concat(bims, ignore_index=True)

    sumstats_combine = pd.read_csv(sumstatslst,
                                    header=None,
                                    names=['sumstats', 'weights'],
                                    delim_whitespace=True)                       
    Nsumstats = len(sumstats_combine)

    df_sumstats1 = pd.read_csv(sumstats1, delim_whitespace=True)

    df_sumstats_combine = [pd.read_csv(file, delim_whitespace=True)
                            for file in sumstats_combine['sumstats']]

    # rename cols
    bim.rename(columns={'A1': 'A1_ref', 'A2': 'A2_ref'}, inplace=True)
    df_sumstats1.rename(columns={'A1': 'A1_x', 'A2': 'A2_x', 'N': 'N_x', 'Z':'Z_x'}, 
        inplace=True)
    for i in range(Nsumstats):
        df_sumstats_combine[i].rename(columns={'A1': 'A1_{}'.format(i), 
            'A2': 'A2_{}'.format(i), 'N': 'N_{}'.format(i), 'Z': 'Z_{}'.format(i)}, 
            inplace=True)

    # take overlap between output and ref genotype files

    df = pd.merge(bim, df_sumstats1, on=['SNP'])

    for i in range(Nsumstats):
        df = df.merge(df_sumstats_combine[i], on=['SNP'])

    # flip sign of z-score for allele reversals
    allign_alleles(df, Nsumstats)
    df = df.loc[np.logical_not(df.SNP.duplicated(keep=False))]
    Z_y = np.zeros(len(df), dtype=float)
    if Nlst is None:
        df_Nlst = pd.DataFrame({'N':[np.max(df['N_{}'.format(i)]) 
            for i in range(Nsumstats)]})
    else:
        df_Nlst = pd.read_csv(Nlst, header=None, names=['N'], delim_whitespace=True)

    for i in range(Nsumstats):
        Z_y += df['Z_{}'.format(i)] * sumstats_combine['weights'].iloc[i] / np.sqrt(df_Nlst['N'].iloc[i])
    
    df.loc[:, 'Z_y'] = Z_y

    if N1 is None:
        N1 = df['N_x'].max()

    return (df.loc[:, ['CHR', 'SNP', 'Z_x', 'Z_y']],
            N1)
 