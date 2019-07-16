import pandas as pd
import numpy as np
import argparse
import pdb

def binPred(cts_result_dir,ldcts_dir,annot_prefix,pip_obj,chrom,result_dir):
    '''
    cts_result_dir is the directory that contains all the cell_type_results files for all relevant datasets.
    ldcts_dir is the directory that contains all relevant ldcts files.
    annot_prefix is the prefix of some version of baseline annotations.
    pip_obj is a PIP object that specifies the pip file, pip column name, SNP column name etc.
    chrom is the chromosome we want to exclude.
    result_dir is the directory we want to store the results in.
    '''
    cts_annotdf = getCtsAnnots(cts_result_dir,ldcts_dir)
