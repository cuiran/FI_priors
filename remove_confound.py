#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import pdb
import glob
import sys
import re
from binned_pred import *

def binpred_no_confound(chrom,pip_file,num_bins,ypred_dir):
    # pip_file relevant columns: prob (pip) and v (SNP)
    pipdf = pd.read_csv(pip_file,delim_whitespace=True)
    files = glob.glob(ypred_dir+'*pred'+chrom+'.'+chrom+'.ypred')
    if len(files)!=21:
        print('Number of ypred files doesn not match 21',len(files))
        sys.exit(1)
    prefixes = ['.'.join(x.split('.')[:-2])+'.' for x in files]
    l = list()
    for f in prefixes:
        binchrom = re.search('_bin(.*)_',f).group(1)
        print('Computing binned predictions with binning chromosome '+binchrom)
        pipdfchr = pipdf[pipdf['chromosome']==int(binchrom)]
        if len(pipdfchr)==0:
            print('No PIP information for chromosome '+binchrom)
            continue
        ypreddf = pd.read_csv(f+chrom+'.ypred',delim_whitespace=True)
        binypreddf = pd.read_csv(f+binchrom+'.ypred',delim_whitespace=True)
        merged = pd.merge(pipdfchr[['v','prob']],binypreddf,left_on='v',right_on='SNP')
        ybin_stdized = min_max_scale(merged['YPRED'].values)
        ypred_stdized = min_max_scale(ypreddf['YPRED'].values)
        merged[binchrom] = ybin_stdized
        ypreddf[chrom] = ypred_stdized
        print('Calculating predictions in bins for binning chromosome '+binchrom)
        bin_pred,cutoffs = get_bin_pred(merged,binchrom,num_bins,'prob','equally-sized')
        predinbindf = pd.DataFrame(data=bin_pred,columns=['PRED_IN_BIN'])
        predinbindf.to_csv(f+str(num_bins)+'bins.predinbins',sep='\t',index=False)
        cutoffdf = pd.DataFrame(data=cutoffs,columns=['CUTOFF'])
        cutoffdf.to_csv(f+str(num_bins)+'bins.cutoffs',sep='\t',index=False)
        print('Applying binned predictions on chromosome '+chrom)
        #ypreddf[str(chrom)+'_binpred'] = ypreddf.apply(lambda row:assign_binpred(row,chrom,bin_pred,cutoffs),axis=1)
        ypreddf = assign_binpred2(ypreddf,chrom,bin_pred,cutoffs)
        ypreddf.rename(columns={chrom:'ypred_stdized','binpred':'ybinpred'},inplace=True)
        ypreddf.to_csv(f+str(num_bins)+'bins.ybinpred',sep='\t',index=False)
        l.append(ypreddf['ybinpred'].tolist())
    meanbinpred = np.mean(np.array(l), axis=0)
    meanbinpreddf = pd.DataFrame(data=None)
    meanbinpreddf['SNP'] = ypreddf['SNP']
    meanbinpreddf['mean_binpred'] = meanbinpred
    fnameprefix = '_'.join(f.split('_')[:-2])
    fname = fnameprefix+'_pred'+chrom+'.'+str(num_bins)+'bins.meanbinpred'
    meanbinpreddf.to_csv(fname,sep='\t',index=False)
    return
        

def filelist(directory,chrom):
    # get all the file names with the chrom as a binning or prediction chromosome
    # check the length of each set of files should be 21
    # output one list of 42 file names
    binlist = glob.glob(directory+'*bin'+chrom+'_*.coef')
    if len(binlist)!=21:
        print('Number of files with binning chromosome '+chrom+' does not match 21', len(binlist))
        sys.exit(1)
    predlist = glob.glob(directory+'*pred'+chrom+'.coef')
    if len(predlist)!=21:
        print('Number of files with prediction chromosome '+chrom+' does not match 21', len(predlist))
        sys.exit(1)
    return binlist+predlist

def ypred_no_confound(annot_prefix,chrom,coef_dir,result_dir,conf_names):
    # annot_prefix is the prefix to the annotation files with '.' included
    # conf_names is the names of the confounding annotations
    print('reading in annotation for chromosome '+chrom)
    annot_df = pd.read_csv(annot_prefix+chrom+'.annot.gz',delim_whitespace=True)
    col_names = annot_df.columns
    info_cols = ['CHR','BP','SNP','CM']
    annot_names = [x for x in col_names if x not in info_cols]
    keep = [x for x in col_names if x not in conf_names+info_cols]
    annot_noconf = annot_df.loc[:,keep]
    # fill nan with 0 (theres a bug in creating the baselineLF.22.annot.gz code that's unsolved)
    if chrom == '22':
        annot_noconf.fillna(0,inplace=True)
    coef_list = filelist(coef_dir,chrom)
    for f in coef_list:
        print('processing file '+f)
        coef_df = pd.read_csv(f,delim_whitespace=True)
        coef_df.index = annot_names
        coef_noconf = coef_df.loc[keep,:]
        newpred_df = dotprod(annot_df['SNP'].tolist(),annot_noconf,coef_noconf)
        fname = get_fname_noext(f)
        newpred_df.to_csv(result_dir+fname+'.'+chrom+'.ypred',sep='\t',index=False)
    return

def get_fname_noext(fname):
    # given a file name, this function strips the directory and extension information
    nodir = fname.split('/')[-1]
    noext = '.'.join(nodir.split('.')[:-1])
    return noext

def dotprod(snplist,annot_noconf,coef_noconf):
    # dot product between annotation and coef
    coefs = coef_noconf.values
    annot = annot_noconf.values
    preddf = pd.DataFrame(data=annot.dot(coefs),columns=['YPRED'])
    preddf['SNP'] = snplist
    return preddf

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom',type=str)
    parser.add_argument('--coef-dir')
    parser.add_argument('--result-dir')
    parser.add_argument('--annot-prefix')
    parser.add_argument('--compute-ypred',action='store_true')
    parser.add_argument('--compute-binpred',action='store_true')
    parser.add_argument('--pip-file')
    parser.add_argument('--num-bins',type=int)
    parser.add_argument('--ypred-dir')
    parser.add_argument('--BLD',action='store_true')
    args = parser.parse_args()

    print('assign list of confoundings to conf_names')

    conf_names_BLF = ['MAFbin_lowfreq_1', 'MAFbin_lowfreq_2', 'MAFbin_lowfreq_3', 'MAFbin_lowfreq_4', 'MAFbin_lowfreq_5', 'MAFbin_lowfreq_6', 'MAFbin_lowfreq_7', 'MAFbin_lowfreq_8', 'MAFbin_lowfreq_9', 'MAFbin_lowfreq_10', 'MAFbin_frequent_1', 'MAFbin_frequent_2', 'MAFbin_frequent_3', 'MAFbin_frequent_4', 'MAFbin_frequent_5', 'MAFbin_frequent_6', 'MAFbin_frequent_7', 'MAFbin_frequent_8', 'MAFbin_frequent_9', 'MAFbin_frequent_10', 'MAF_Adj_Predicted_Allele_Age_common', 'MAF_Adj_LLD_AFR_lowfreq', 'MAF_Adj_LLD_AFR_common', 'MAF_Adj_ASMC_lowfreq', 'MAF_Adj_ASMC_common']
    conf_names_BLD = ['MAFbin1', 'MAFbin2', 'MAFbin3', 'MAFbin4', 'MAFbin5', 'MAFbin6', 'MAFbin7', 'MAFbin8', 'MAFbin9', 'MAFbin10', 'MAF_Adj_Predicted_Allele_Age', 'MAF_Adj_LLD_AFR', 'MAF_Adj_ASMC']

    if args.compute_ypred:
        print('Compute ypred without MAF and LD confoundings')
        print('Annot prefix',args.annot_prefix)
        print('chrom', args.chrom)
        print('coefficient directory',args.coef_dir)
        print('result directory',args.result_dir)
        if args.BLD:
            ypred_no_confound(args.annot_prefix,args.chrom,args.coef_dir,args.result_dir,conf_names_BLD)
        else:
            ypred_no_confound(args.annot_prefix,args.chrom,args.coef_dir,args.result_dir,conf_names_BLF)
    elif args.compute_binpred:
        print('Computing binpred without MAF and LD confoundings')
        print('chrom: '+args.chrom)
        print('pip file: '+args.pip_file)
        print('number of bins: '+str(args.num_bins))
        print('ypred directory: '+str(args.ypred_dir))
        binpred_no_confound(args.chrom,args.pip_file,args.num_bins,args.ypred_dir)
