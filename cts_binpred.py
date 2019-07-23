#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
import pdb
from functools import partial

def binPredCV(args):
    feature = Feature(ref_annot=Prefix('ref_annot','in'),
                      cts_annot=Prefix('cts_annot','in'),
                      cts_result_folder=args.cts_result_folder,
                      ldcts_folder=args.ldcts_folder,
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
        ypred_binchrom = model.predict(merge(feature,target,binchrom),remove_confounders=True)
        ypred_predchrom = model.predict(feature.chrom(predchrom),remove_confounders=True)
        binpred.compute_binpred(ypred_binchrom,ypred_predchrom)
    binpred.compute_meanbinpred()
    return


def Prefix(name,in_out):
    if name=='feature' and in_out=='out':
        return args.feature_output_prefix


def check_shape(df1,df2,axis):
    '''Checking if the shape of two dfs match
    along the specified axis
    '''
    if df1.empty or df2.empty:
        if df1.empty:
            warnings.warn('Dataframe in the first argument is empty')
        if df2.empty:
            warnings.warn('Dataframe in the second argument is empty')
        return True
    if not df1.empty and not df2.empty:
        shape1 = df1.shape
        shape2 = df2.shape
        if axis=='row':
            if shape1[0]==shape2[0]:
                return True
            else:
                return False
        elif axis=='column':
            if shape1[1]==shape2[1]:
                return True
            else:
                return False


def containList(l1,l2):
    '''Checking if list 1 contains in list 2
    '''
    return all(x in l2 for x in l1)


class Feature:

    def __init__(self,ref_annot_prefix=None,cts_annot_prefix=None,thin=False,
                 cts_result_folder=None,ldcts_folder=None,
                 info_colnames=['CHR','SNP','CM','BP']):
        self.ref_annot_prefix = ref_annot_prefix
        self.cts_annot_prefix = cts_annot_prefix
        self.cts_result_folder = cts_result_folder # folder for cell_type_result.txt files
        self.ldcts_folder = ldcts_folder # folder for ldcts files
        self.info_colnames = info_colnames
        self.thin = thin # indicating if cts annotations are thin annotations

    @memoize
    def chromosome(self,chrom):
        '''Outputs the feature dataframe for chromosome=chrom
        The dataframe should include the info columns
        '''
        print('Getting features for chromosome {}'.format(chrom))
        if chrom not in [x for x in range(1,23)]:
            raise Exception('{} is not a valid chromosome number'.format(chrom))
        else:
            ref_df = self.ref_annot_df(chrom)
            cts_df = self.cts_annot_df(chrom)
            if not check_shape(ref_df,cts_df,'row'):
                raise Exception('Shape of reference annotations doesn not match cell type specific annotations.\n
                                Reference annotation shape {},\n
                                cell type specific annotation shape {}'.format(ref_df.shape,cts_df.shape))
            else:
                if self.thin:
                    return pd.concat([ref_df,cts_df],axis=1)
                else:
                    if containList(self.info_colnames,cts_df.columns.tolist()):
                        cts_df.drop(self.info_colnames,axis=1,inplace=True)
                        return pd.concat([ref_df,cts_df],axis=1)
                    else:
                        raise Exception('Cell type specific annotations does not contain all info column names')

    @memoize
    def ref_annot_df(self,chrom):
        '''Outputs the reference annotation dataframe with info columns
        '''
        if self.ref_annot_prefix is None:
            return pd.DataFrame(None)
        else:
            chrom = str(chrom)
            return pd.read_csv(self.ref_annot_prefix+chrom+'.annot.gz',delim_whitespace=True)

    @memoize
    def cts_annot_df(self,chrom):
        '''Outputs the cell type specific annotation dataframe 
        Depending if the cts annots are thin or not, it may or may not 
        include the info columns.
        '''
        if cts_annot_prefix is None:
            return get_cts_annot_df(self,chrom)
        else:
            chrom = str(chrom)
            return pd.read_csv(self.cts_annot_prefix+chrom+'.annot.gz',delim_whitespace=True)


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


class memoize(object):
    """cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

