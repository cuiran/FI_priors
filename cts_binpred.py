#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
import pdb
from functools import partial
import data
import model

def binPredCV(args):
    feature = data.Feature(ref_annot=data.Prefix('ref_annot','in'),
                      cts_annot=data.Prefix('cts_annot','in'),
                      cts_result_folder=args.cts_result_folder,
                      ldcts_folder=args.ldcts_folder,
                      output_prefix=data.Prefix('feature','out'))
    target = data.Target(target_file=args.target_file,
                    target_colname=args.target_colname)
    predchrom = args.chrom
    binchroms = [x for x in range(1,23) if x != int(predchrom)]
    binpred = BinPred(predchrom=predchrom,
                      binchroms=binchroms,
                      num_bins=args.num_bins,
                      bin_method=args.bin_method
                      output_prefix=data.Prefix('binpred','out'))
    for binchrom in binchroms:
        m = model.Model(method=args.method,
                      exclude_chrs=[predchrom,binchrom],
                      confounders=args.confounder_names,
                      output_prefix=data.Prefix('model','out'))
        m.fit(feature,target)
        ypred_binchrom = m.predict(merge(feature,target,binchrom),remove_confounders=True)
        ypred_predchrom = m.predict(feature.chrom(predchrom),remove_confounders=True)
        binpred.compute_binpred(ypred_binchrom,ypred_predchrom)
    binpred.compute_meanbinpred()
    return


class BinPred:

    def __init__(self,predchrom,binchroms,num_bins=5,bin_method='equally-sized'):
        self.predchrom = predchrom # integer
        self.binchroms = binchroms # list (can be list of one element)
        self.num_bins = num_bins
        self.bin_method = bin_method # equally-sized or equally-spaced

