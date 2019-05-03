import pandas as pd
import numpy as np
import pdb

def make_bed(ssfile,snps,outfile,windowsize):
    # assume sumstats file has 'P' column
    # use significance cutoff 5e-8
    # snps is a csv file with columns ['CHR','SNP','CM','BP','A1','A2']
    # create bed file with three columns: chr*, startbp, endbp. Saving without the column names
    ssdf = pd.read_csv(ssfile,delim_whitespace=True)
    sigdf = ssdf[ssdf['P']<=5e-8]
    sigdf_bp = attach_bp(sigdf,snps,ssfile)
    startbp = sigdf_bp['BP'].values - windowsize
    startbp = np.fmax(0,startbp)
    endbp = sigdf_bp['BP'].values + windowsize
    chrom = ['chr'+str(i) for i in sigdf_bp['CHR'].values]
    beddf = pd.DataFrame({'CHR':chrom,'START_BP':startbp,'END_BP':endbp})
    pdb.set_trace()
    beddf.to_csv(outfile,index=False,sep='\t',header=False)
    return

def attach_bp(df,snps,outfile_prefix):
    # df will be a formatted sumstats df, it has SNP column but no BP or CHR information
    # snps is a csv file with columns ['CHR','SNP','CM','BP','A1','A2']
    bp = pd.read_csv(snps,delim_whitespace=True)
    merged = pd.merge(df,bp[['CHR','SNP','BP']],on=['SNP'])
    # drop duplicated rows
    merged_nodup = merged.drop_duplicates()
    merged_nodup.to_csv(outfile_prefix+'_BP',sep='\t',index=False)
    return merged_nodup
   

if __name__=='__main__':
    make_bed('/n/groups/price/ran/FI_priors/data/sig/UKB_460K.body_HEIGHTz.P5e-8.sumstats','/n/groups/price/ran/high-dim-sldsc/0.data/UKBB/bim_files/UKBB.snplist','/n/groups/price/ran/FI_priors/data/bedfiles/UKB_460K.body_HEIGHTz.sigP.wind1M.bed',500000) 
