#!/usr/bin/env python
'''
multi-trait LD score regression

MTLDSC

Created on 2021-10-11

@author: Yiliang Zhang
'''


import argparse, os.path, sys
from prep import prep
from ldsc_thin import ldscore
from calculate import calculate
import pandas as pd
import numpy as np

try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.drop_duplicates(subset='A')
except TypeError:
    raise ImportError('GENJI requires pandas version > 0.15.2')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
pd.set_option('max_colwidth',1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)

# returns whether the parent directory of path exists
def parent_dir_exists(path):
    return os.path.exists(os.path.abspath(os.path.join(path, os.pardir)))


def pipeline(args):
    pd.options.mode.chained_assignment = None

    # Sanity check args
    if not parent_dir_exists(args.out):
        raise ValueError('--out flag points to an invalid path.')

    print('Preparing files for analysis...')
    gwas_snps, N1, intercept = prep(args.bfile, args.sumstats1, args.sumstatslst, args.N1, args.Nlst)
    print('Calculating LD scores...')
    ld_scores = ldscore(args.bfile, gwas_snps)
    print('Calculating correlation...')
    out = calculate(gwas_snps, ld_scores, N1, intercept)
    out.to_csv(args.out, sep=' ', na_rep='NA', index=False)


parser = argparse.ArgumentParser()

parser.add_argument('sumstats1',
    help='The first sumstats file.')
parser.add_argument('sumstatslst',
    help='A dataframe of two columns. The first column is the list of sumstats need to be combined. '
    'The second column is the list of the corresponding weights in the combination.')

parser.add_argument('--bfile', required=True, type=str,
    help='Prefix for Plink .bed/.bim/.fam file.')
parser.add_argument('--N1', type=int,
    help='N of the sumstats1 file. If not provided, this value will be inferred '
    'from the sumstats1 arg.')
parser.add_argument('--Nlst', type=int,
    help='The list of N of the sumstats in sumstatslst. If not provided, this value will be inferred '
    'from the sumstats.')

parser.add_argument('--out', required=True, type=str,
    help='Location to output results.')

if __name__ == '__main__':
    pipeline(parser.parse_args())
