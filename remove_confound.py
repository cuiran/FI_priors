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
    files = glob.glob(ypred_dir+'*pred'+chrom+'.*')
    prefixes = list(set(['.'.join(x.split('.')[:-2])+'.' for x in files]))
    l = list()
    for f in prefixes:
        binchrom = re.search('bin(.*)_').group(1)
        binypreddf = pd.read_csv(f+binchrom+'.ypred',delim_whitespace=True)
        ypreddf = pd.read_csv(f+chrom+'.ypred',delim_whitespace=True)
        pipdfchr = pipdf[pipdf['chromosome']==str(binchrom)]
        merged = pd.merge(pipdfchr[['v','prob']],ypreddf,left_on='v',right_on='SNP')
        ybin_stdized = min_max_scale(merged['YPRED'].values)
        ypred_stdized = min_max_scale(ypreddf['YPRED'].values)
        merged['YPRED_stdized'] = ybin_stdized
        ypreddf[chrom] = ypred_stdized
        print('Calculating predictions in bins for binning chromosome '+binchrom)
        bin_pred,cutoffs = get_bin_pred(merged,binchrom,num_bins,'prob','equally-sized')
        predinbindf = pd.DataFrame(data=bin_pred,columns=['PRED_IN_BIN'])
        predinbindf.to_csv(f+str(num_bins)+'bins.predinbins',sep='\t',index=False)
        cutoffdf = pd.DataFrame(data=cutoffs,columns=['CUTOFF'])
        cutoffdf.to_csv(f+str(num_bins)+'bins.cutoffs',sep='\t',index=False)
        print('Applying binned predictions on chromosome '+chrom)
        ypreddf[chrom+'_binpred'] = ypreddf.apply(lambda row:assign_binpred(row,pred_chrom,bin_pred,cutoffs),axis=1)
        ypreddf.rename(columns={chrom:'ypred_stdized',chrom+'_binpred':'ybinpred'},inplace=True)
        ypreddf.to_csv(f+str(num_bins)+'bins.ybinpred',sep='\t',index=False)
        l.append(ypreddf['ybinpred'].tolist())
    meanbinpred = np.mean(np.array(l), axis=0)
    meanbinpreddf = pd.DataFrame(data=meanbinpred,columns=['mean_binpred'])
    meanbinpreddf['SNP'] = ypreddf['SNP']
    meanbinpreddf.to_csv(f+str(num_bins)+'bins.meanbinpred',sep='\t',index=False)
    return
        

def filelist(directory,chrom):
    # get all the file names with the chrom as a binning or prediction chromosome
    # check the length of each set of files should be 21
    # output one list of 42 file names
    binlist = glob.glob(directory+'*bin'+chrom+'_*')
    if len(binlist)!=21:
        print('Number of files with binning chromosome '+chrom+' does not match 21', len(binlist))
        sys.exit(1)
    predlist = glob.glob(directory+'*pred'+chrom+'.*')
    if len(predlist)!=21:
        print('Number of files with prediction chromosome '+chrom+' does not match 21', len(predlist))
        sys.exit(1)
    return binlist+predlist

def ypred_no_confound(pip_file,num_bins,annot_prefix,chrom,coef_dir,result_dir,conf_names):
    # annot_prefix is the prefix to the annotation files with '.' included
    # conf_names is the names of the confounding annotations
    print('reading in annotation for chromosome '+chrom)
    annot_df = pd.read_csv(annot_prefix+chrom+'.annot.gz',delim_whitespace=True)
    col_names = annot_df.columns
    info_cols = ['CHR','BP','SNP','CM']
    annot_names = [x for x in col_names if x not in info_cols]
    keep = [x for x in col_names if x not in conf_names+info_cols]
    annot_noconf = annot_df.loc[:,keep]
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
    parser.add_argument('--coef_dir')
    parser.add_argument('--result_dir')
    parser.add_argument('--annot_prefix')
    parser.add_argument('--compute_ypred',action='store_true')
    args = parser.parse_args()

    conf_names = ['MAFbin_lowfreq_1', 'MAFbin_lowfreq_2', 'MAFbin_lowfreq_3', 'MAFbin_lowfreq_4', 'MAFbin_lowfreq_5', 'MAFbin_lowfreq_6', 'MAFbin_lowfreq_7', 'MAFbin_lowfreq_8', 'MAFbin_lowfreq_9', 'MAFbin_lowfreq_10', 'MAFbin_frequent_1', 'MAFbin_frequent_2', 'MAFbin_frequent_3', 'MAFbin_frequent_4', 'MAFbin_frequent_5', 'MAFbin_frequent_6', 'MAFbin_frequent_7', 'MAFbin_frequent_8', 'MAFbin_frequent_9', 'MAFbin_frequent_10', 'MAF_Adj_Predicted_Allele_Age_common', 'MAF_Adj_LLD_AFR_lowfreq', 'MAF_Adj_LLD_AFR_common', 'MAF_Adj_ASMC_lowfreq', 'MAF_Adj_ASMC_common']
    if args.compute_ypred:
        ypred_no_confound(args.annot_prefix,args.chrom,args.coef_dir,args.result_dir,conf_names)
