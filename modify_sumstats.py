#!/usr/bin/env python

import pandas as pd
import pdb
import argparse

def eo_binpred(bin_eo,pred_chrom,ss_prefix,output_file):
    '''
    This function will create a summary statistics file that only contain SNPs in the binning chromosomes 
    '''
    trainchroms = [x for x in range(1,23) if x!=int(pred_chrom)]
    if bin_eo=='even':
        binchroms = [x for x in trainchroms if x%2==0]
    elif bin_eo=='odd':
        binchroms = [x for x in trainchroms if x%2!=0]
    files = [ss_prefix+str(chrom)+'.sumstats' for chrom in binchroms]
    dfs = [pd.read_csv(f,delim_whitespace=True) for f in files]
    df = pd.concat(dfs,axis=0,sort=False)
    df.to_csv(output_file,sep='\t',index=False)
    return

def chrsep_ss(ss_file,bim_prefix,out_prefix):
    '''
    This function will split the summary stats file into chromosome seperated files 
    according to the bim files.

    Bim file column order must be: ['CHR','SNP','CM','BP','A1','A2']
    '''
    for i in range(1,23):
        chrom = str(i)
        print('Processing chromosome '+chrom)
        bimdf = pd.read_csv(bim_prefix+chrom+'.bim',delim_whitespace=True,header=None)
        ssdf = pd.read_csv(ss_file,delim_whitespace=True)
        bimdf.columns=['CHR','SNP','CM','BP','A1','A2']
        merged = pd.merge(ssdf,bimdf[['SNP','A1','A2']],on=['SNP','A1','A2'])
        print('Saving chromosome separated sumstat file to '+out_prefix+chrom+'.sumstats')
        merged.to_csv(out_prefix+chrom+'.sumstats',sep='\t',index=False)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chr-sep',action='store_true',help='Create chromosome separated files')
    parser.add_argument('--eo-split',action='store_true',help='Create sumstats based on specified even odd split')
    parser.add_argument('--ss',help='Summary Statistics')
    parser.add_argument('--output-prefix')
    parser.add_argument('--bim-prefix')
    parser.add_argument('--bin-eo')
    parser.add_argument('--pred-chrom')
    parser.add_argument('--ss-prefix')
    parser.add_argument('--output-file')
    args = parser.parse_args()

    if args.chr_sep:
        chrsep_ss(args.ss,args.bim_prefix,args.output_prefix)
    elif args.eo_split:
        eo_binpred(args.bin_eo,args.pred_chrom,args.ss_prefix,args.output_file)
