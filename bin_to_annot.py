import pandas as pd
import numpy as np
import argparse
import pdb
import glob

def bin_to_annot(bim_file, cutoff_file, ypred_file, ref_file, chrom, annot_prefix):
    '''
    This function makes annotation files based on ybinpred files.
    The annotations will be disjoint.

    bim_file is the bim file name according to which the ybinpred file was based on.

    cutoff_file is the name of the file which stores the cutoff points to the binning
    the cutoff values are based on standardized ypred values, therefore the min cutoff is always 0
    and the max is always 1

    ypred_file is the name of the file which stores the prediction values, 
    it's assumed that this file only has one column named 'YPRED'

    ref_file contains the SNP information for the predicted SNPs in ypred_file

    chrom is the chromosome of interest

    annot_prefix is the prefix of the resulting annotation files. 
    '''
    bimdf = pd.read_csv(bim_file,delim_whitespace=True,header=None)
    bimdf.columns = ['CHR','SNP','CM','BP','A1','A2']
    ydf = pd.read_csv(ypred_file,delim_whitespace=True)
    refdf = pd.read_parquet(ref_file,engine='pyarrow')
    ydf['SNP'] = refdf['SNP']
    print('Merging dataframes')
    df = pd.merge(bimdf,ydf[['SNP','YPRED']],how='left',on=['SNP'])
    df['ypred'] = df['YPRED'].copy()
    df['YPRED'].fillna(value=0.0,inplace=True)
    df['YPRED'] = max_min_scale(df['YPRED'].values)
    df.drop_duplicates(inplace=True)
    df['ypred'][~df['ypred'].isnull()]=1.0
    df['YPRED'] = df['YPRED']*df['ypred']
    df.fillna(value=-1.0,inplace=True)
    cuts = pd.read_csv(cutoff_file,delim_whitespace=True).values.ravel()
    cuts[-1] = 1.001
    for i in range(1,len(cuts)):
        print('Making annotations for bin '+str(i))
        df['BIN'+str(i)] = np.where((df['YPRED']>=cuts[i-1])&(df['YPRED']<cuts[i]),'1.0','0.0')
    print('Writing annotation to file')
    df.drop(['A1','A2','YPRED','ypred'],axis=1,inplace=True)
    df.to_csv(annot_prefix+chrom+'.annot.gz',sep='\t',index=False,compression='gzip')
    return

def bin_to_annot_multichrom(bim_prefix,cutoff_file,ypred_file,annot_prefix):
    '''
    bim_prefix is the prefix for bim files, stopping right before the chromosome number,
    these files has no header, assign column names ['CHR','RSID','CM','BP','A1','A2']

    cutoff_file is the file that stores cutoff points for the standardized YPRED values, with one column: 'CUTOFF'

    ypred_file is the file that stores YPRED values for the binning chromosomes, columns=['CHR','BP,'SNP','CM','pip','YPRED']

    annot_prefix is the prefix of the resulting annotation files.
    '''
    annot_chroms,predf = get_chroms(ypred_file)
    for i in annot_chroms:
        chrom = str(i)
        print('Annotating chromosome '+chrom)
        ydf = predf[predf['CHR']==i].copy()
        ydf['scaled_ypred'] = max_min_scale(ydf['YPRED'].values)
        bimdf = pd.read_csv(bim_prefix+chrom+'.bim',delim_whitespace=True,header=None)
        bimdf.columns=['CHR','RSID','CM','BP','A1','A2']
        bimdf['SNP'] = bimdf['CHR'].astype(str)+':'+bimdf['BP'].astype(str)+':'+bimdf['A1']+':'+bimdf['A2']
        df = pd.merge(bimdf,ydf[['SNP','scaled_ypred']],how='left',on=['SNP'])
        df.fillna(value=-1.0,inplace=True)
        cuts = pd.read_csv(cutoff_file,delim_whitespace=True).values.ravel()
        cuts[-1] = 1.001
        for j in range(1,len(cuts)):
            print('Making annotations for bin '+str(j))
            df['BIN'+str(j)] = np.where((df['scaled_ypred']>=cuts[j-1])&(df['scaled_ypred']<cuts[j]),'1.0','0.0')
        print('Writing annotation to file')
        df.drop(['A1','A2','scaled_ypred','SNP'],axis=1,inplace=True)
        df.rename(columns={"RSID":"SNP"},inplace=True)
        df.to_csv(annot_prefix+chrom+'.annot.gz',sep='\t',index=False,compression='gzip')
    return
        
        

def get_chroms(f):
    '''
    f is the file name of interest.
    This file should contain column names, one of which should be 'CHR'

    This function outputs the different chromosomes numbers that are in this file, as well as the dataframe from the file
    '''
    df = pd.read_csv(f,delim_whitespace=True)
    return df['CHR'].unique(),df
    

def create_zero_annots(bim_prefix,annot_prefix,num_bins):
    '''
    This function creates zero annotations for all chromosomes except for those which already has an annotation with the same prefix

    num_bins is the number of bins in the annotation.
    '''
    files = glob.glob(annot_prefix+'*.annot.gz')
    exist_chroms = [s.split('.')[-3] for s in files]
    exist_chroms = [int(x) for x in exist_chroms]
    chroms = [x for x in range(1,23) if x not in exist_chroms]
    for chrom in chroms:
        print('Creating zero annotation for chromosome '+str(chrom))
        bimdf = pd.read_csv(bim_prefix+str(chrom)+'.bim',delim_whitespace=True,header=None)
        bimdf.columns=['CHR','SNP','CM','BP','A1','A2']
        for i in range(1,int(num_bins)+1):
            bimdf['BIN'+str(i)] = 0.0
        bimdf.drop(['A1','A2'],axis=1,inplace=True)
        bimdf.to_csv(annot_prefix+str(chrom)+'.annot.gz',sep='\t',index=False,compression='gzip')
    return

def max_min_scale(a):
    '''
    Scaling array a to range [0,1] using the fomula below
    (a-min(a))/(max(a)-min(a))
    '''
    mi = float(min(a))
    ma = float(max(a))
    diff = ma-mi
    return (a-mi)/diff

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-to-annot',action='store_true')
    parser.add_argument('--zero-annots',action='store_true')
    parser.add_argument('--single-chrom',action='store_true')
    parser.add_argument('--multi-chrom',action='store_true')
    parser.add_argument('--bim-file',help='one bim file')
    parser.add_argument('--cutoff-file')
    parser.add_argument('--ybinpred-file')
    parser.add_argument('--ypred-file')
    parser.add_argument('--ref-file')
    parser.add_argument('--chrom')
    parser.add_argument('--annot-prefix')
    parser.add_argument('--bim-prefix',help='prefix of many bim files')
    parser.add_argument('--num-bins',type=int)
    args = parser.parse_args()

    if args.bin_to_annot:
        bin_to_annot(args.bim_file,args.cutoff_file,args.ypred_file,args.ref_file,args.chrom,args.annot_prefix)
    elif args.zero_annots:
        create_zero_annots(args.bim_prefix,args.annot_prefix,args.num_bins)
    elif args.multi_chrom:
        bin_to_annot_multichrom(args.bim_prefix,args.cutoff_file,args.ypred_file,args.annot_prefix)
