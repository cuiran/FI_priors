#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import pdb

def binPredCV(args):
    feature = Feature(ref_annot=Prefix('ref_annot','in'),
                      cts_annot=Prefix('cts_annot','in'),
                      output_prefix=Prefix('feature','out'))
    target = Target(target_file=args.target_file,
                    target_colname=args.target_colname)
    predchrom = args.chrom
    binchroms = [x for x in range(1,23) if x != int(predchrom)]
    binpred =BinPred(predchrom=predchrom,
                     binchroms=binchroms,
                     num_bins=args.num_bins,
                     bin_method=args.bin_method
                     output_prefix=Prefix('binpred','out'))
    for binchrom in binchroms:
        model = Model(method=args.method,
                      exclude_chrs=[predchrom,binchrom],
                      confounders=args.confounder_names,
                      output_prefix=Prefix('model','out'))
        model.fit(feature,target)
        ypred_binchrom = model.predict(feature.binchrom.withpip,remove_confounders=True)
        ypred_predchrom = model.predict(feature.predchrom.all,remove_confounders=True)
        binpred.compute_binpred(ypred_binchrom,ypred_predchrom)
    binpred.compute_meanbinpred()
    return


def Prefix(name,in_out):
    if name=='feature' and in_out=='out':
        return args.feature_output_prefix


class Feature:

    def __init__(self,ref_annot,cts_annot):
        self.ref_annot = ref_annot # reference annotation prefix
        self.cts_annot = cts_annot # cell type specific annotation prefix


class Target:

    def __init__(self,target_file,target_colname='prob'):
        self.target_file = target_file # usually PIP file with all fine-mapped SNPs
        self.target_colname = target_colname # usually the column name for PIP


class BinPred:

    def __init__(self,predchrom,binchroms,num_bins=5,bin_method='equally-sized'):
        self.predchrom = predchrom # integer
        self.binchroms = binchroms # list (can be list of one element)
        self.num_bins = num_bins
        self.bin_method = bin_method # equally-sized or equally-spaced


class Model:

    def __init__(self,method='OLS',exclude_chrs,confounders,output_prefix)
