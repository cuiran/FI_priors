import pandas as pd
import numpy as np
import pdb
import argparse
import os

def chr_sep(finemap_parquet_file,out_prefix):
    '''
    The initial step of creating chromosome seperated target files from raw data. 
    '''
    f = finemap_parquet_file
    print('Reading in parquet file')
    df = pd.read_parquet(f,engine='pyarrow')
    print('Resetting index')
    df.reset_index(inplace=True)
    print('Create chromosome separated files')
    for i in range(1,23):
        dfchr = df[df['CHR']==i]
        dfchr.columns = ['CHR','BP','SNP','A1','A2','info','pip']
        dfchr.to_parquet(out_prefix+str(i)+'.parquet',engine='pyarrow')
        print('Finished seperating out chromosome '+str(i))
    return

def match_snps(target_prefix,annot_prefix,annot_ref_prefix,target_output_prefix,annot_output_prefix,col_name,recompute):
    '''
    Merge target dataframe with annotation dataframe on 'SNP' column.
    Assuming target data format to be .parquet, and annotation data format
    to be .annot.gz. 
    
    Annotation dataframe column names must contain:
    ['CHR','BP','SNP','CM'] and the rest of the columns must be exclusively annotation data.
    
    Target dataframe column names must contain 'SNP','BP', and a specified column name for 
    the target values, specified by col_name. col_name must not be the same as any column name
    in the annotation dataframe.

    target_prefix is the prefix of target files, ends right before the chromosome number.
    Target file names must be of the form "target_prefix.CHROM.parquet"

    annot_prefix is the prefix of annotation files, ends right before the chromosoem number.
    Annotation file names must be of the form "annot_prefix.CHROM.annot.gz"

    annot_ref_prefix is the prefix of SNP info files. Provide this only if annotation file doesn't have info columns

    Resulting files will have matching SNPs with same SNP info as in annot.gz files.
    '''
    for i in range(1,23):
        chrom = str(i)
        print('Processing chromosome '+chrom)
        if os.path.isfile(target_output_prefix+chrom+'.parquet') and os.path.isfile(annot_output_prefix+chrom+'.parquet') and not recompute:
            print('Data already merged')
            continue
        yfile = target_prefix+chrom+'.parquet'
        xfile = annot_prefix+chrom+'.parquet'
        ydf = pd.read_parquet(yfile,engine='pyarrow')
        #ydf.drop_duplicates(inplace=True)
        if annot_ref_prefix is not None:
            adf = pd.read_csv(xfile,delim_whitespace=True,header=None)
            rsdf = pd.read_csv(annot_ref_prefix+chrom,delim_whitespace=True)[['CHR','BP','SNP','CM']]
            xdf = pd.concat([rsdf,adf],axis=1)
        else:
            xdf = pd.read_parquet(xfile,engine='pyarrow')
        #xdf.drop_duplicates(inplace=True)
        merged = pd.merge(ydf[['SNP',col_name]],xdf,on=['SNP'])
        M = merged.shape[0]
        print('After merging, '+str(M)+' SNPs remain')
        if not os.path.isfile(target_output_prefix+chrom+'.parquet') or recompute:
            ymerged = merged[['CHR','BP','SNP','CM',col_name]]
            ymerged.to_parquet(target_output_prefix+chrom+'.parquet',engine='pyarrow')
        else:
            print('Merged target file exists at location '+target_output_prefix+chrom+'.parquet')
        if not os.path.isfile(annot_output_prefix+chrom+'.parquet') or recompute:
            xmerged = merged.drop([col_name],axis=1)
            xmerged.columns = xmerged.columns.astype(str)
            xmerged.to_parquet(annot_output_prefix+chrom+'.parquet',engine='pyarrow')
        else:
            print('Merged annotation file exists at location '+annot_output_prefix+chrom+'.parquet')
    return

def leave_one_out(target_prefix,annot_prefix,target_output_prefix,annot_output_prefix,col_name,num_annots,recompute):
    '''
    Leaving out one chromosome and create regression data without SNP info columns.

    Target files are of the form: target_prefix.CHROM.parquet
    Annotation files are of the form: annot_prefix.CHROM.parquet, annotation data must be to the right of SNP info columns.
    
    col_name is the name of the column in the target files which stores the regression target values

    num_annots is the number of annotations

    Output files are PREFIX.CHROM.parquet
    '''
    if all([os.path.isfile(target_output_prefix+str(chrom)+'.parquet') for chrom in range(1,23)]) and all([os.path.isfile(annot_output_prefix+str(chrom)+'.parquet') for chrom in range(1,23)]) and not recompute:
        print('All files exist')
    else: 
        
        print('Reading in target files')
        yfiles = [target_prefix+str(i)+'.parquet' for i in range(1,23)]
        ydfs = [pd.read_parquet(f,engine='pyarrow') for f in yfiles]
        ydf = pd.concat(ydfs)
        print('Reading in annotation files')
        xfiles = [annot_prefix+str(i)+'.parquet' for i in range(1,23)]
        xdfs = [pd.read_parquet(f,engine='pyarrow') for f in xfiles]
        xdf = pd.concat(xdfs)
        for i in range(1,23):
            chrom = str(i)
            if os.path.isfile(target_output_prefix+chrom+'.parquet') and not recompute:
                print('Target file already exists at '+target_output_prefix+chrom+'.parquet')
            else:
                print('Creating regresion target data leaving out chromosome '+chrom)
                ychrdf = ydf[ydf['CHR']!=i]
                M = ychrdf.shape[0]
                print('After leaving out chromosome '+chrom+', '+str(M)+' SNPs remain in target data.')
                targetdf = ychrdf[[col_name]]    
                targetdf.to_parquet(target_output_prefix+chrom+'.parquet',engine='pyarrow')
            if os.path.isfile(annot_output_prefix+chrom+'.parquet') and not recompute:
                print('Annotation file already exists at '+annot_output_prefix+chrom+'.parquet')
            else:
                print('Creating annotation data leaving out chromosome '+chrom)
                xchrdf = xdf[xdf['CHR']!=i]
                M = xchrdf.shape[0]
                print('After leaving out chromosome '+chrom+', '+str(M)+' SNPs remain in annotation data')
                annotdf = xchrdf.iloc[:,-num_annots:]
                annotdf.to_parquet(annot_output_prefix+chrom+'.parquet',engine='pyarrow')
    return 

def match_ref(ref_prefix, ref_suffix, file_prefix, file_suffix, output_prefix, output_suffix, recompute):
    '''
    Match the SNPs in file with reference file. 

    Reference and pre-matched file must be chromosome seperated. Each must have a 'SNP' column.
    '''
    for i in range(1,23):
        chrom = str(i)
        if os.path.isfile(output_prefix+chrom+'.'+output_suffix) and not recompute:
            print('Matched file already exists at '+output_prefix+chrom+'.'+output_suffix)
        else:
            print('Matching SNPs with reference file for chromosome '+chrom)
            if ref_suffix == 'parquet':
                refdf = pd.read_parquet(ref_prefix+chrom+'.'+ref_suffix,engine='pyarrow')
            else:
                refdf = pd.read_csv(ref_prefix+chrom+'.'+ref_suffix,delim_whitespace=True)
            if file_suffix == 'parquet':
                df = pd.read_parquet(file_prefix+chrom+'.'+file_suffix,engine='pyarrow')
            else:
                df = pd.read_csv(file_prefix+chrom+'.'+file_suffix,delim_whitespace=True)
            merged = pd.merge(refdf[['SNP']],df,on=['SNP'])
            print('Saving merged file to '+output_prefix+chrom+'.'+output_suffix)
            if output_suffix == 'parquet':
                merged.to_parquet(output_prefix+chrom+'.'+output_suffix,engine='pyarrow')
            else:
                merged.to_csv(output_prefix+chrom+'.'+output_suffix,sep='\t',index=False)
    return

def discretize(fname,col_name,threshold,output_fname):
    '''
    Discretize the continuous values in col_name of fname according to the threshold value.
    Anything less than the threshold will be labeled 0.
    Anything greater than or equal to the threshold will be labeled 1.
    '''
    if 'parquet' in fname:
        df = pd.read_parquet(fname,engine='pyarrow')
    else:
        df = pd.read_csv(fname,delim_whitespace=True)
    df['label'] = np.where(df['pip']>=threshold,1.0,0.0)
    if 'NOT' in fname:
        df = df[['label']]
    if 'parquet' in output_fname:
        df.to_parquet(output_fname,engine='pyarrow')
    else:
        df.to_csv(output_fname,sep='\t',index=False)
    return
   
def discretize_threebins(fname,col_name,output_fname):
    '''
    This function is meant to discretize values between 0 and 1 into three bins
    bounded by [0.0,0.1,0.25,1.1]
    '''
    if 'parquet' in fname:
        df = pd.read_parquet(fname,engine='pyarrow')
    else:
        df = pd.read_csv(fname,delim_whitespace=True)
    bins = [0.0,0.1,0.25,1.1]
    disc = np.digitize(df[col_name].values,bins) - 1
    df['label'] = disc
    if 'NOT' in fname:
        df = df[['label']]
    if 'parquet' in output_fname:
        df.to_parquet(output_fname,engine='pyarrow')
    else:
        df.to_csv(output_fname,sep='\t',index=False)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match-snps',action='store_true',help='Merge regression data with annotation data on the SNP column. \
                        Assuming target data format is .parquet and annotation data format is .annot.gz.')
    parser.add_argument('--leave-one-out',action='store_true',help='Leave out one chromosome and create 22 files with only regression data.')
    parser.add_argument('--match-ref',action='store_true',help='Match SNP with reference file.')
    parser.add_argument('--discretize',action='store_true',help='Discretize continuous variable.')
    parser.add_argument('--three-bins',action='store_true',help='Used along with --discretize flag. Put continuous values in three bins defined by [0.0,0.1,0.25,1.1]. Input continuous values must be in [0,1]')
    parser.add_argument('--target-prefix',help='Regression target files prefix, chromosome seperated.')
    parser.add_argument('--annot-prefix',help='Annotation files prefix, chromosome seperated.')
    parser.add_argument('--annot-ref-prefix',help='Provide rsid files prefix if annotation files do not have info columns.')
    parser.add_argument('--target-output-prefix',help='Output target files prefix after matching SNPs between target files and annotation files.')
    parser.add_argument('--annot-output-prefix',help='Output annotation files prefix after matching SNPs between target files and annotation files.')
    parser.add_argument('--target-col-name',help='Name of the column that stores the values for regression target')
    parser.add_argument('--recompute',action='store_true',help='Redo the computation, regardless of existing files.')
    parser.add_argument('--num-annots',type=int,help='Number of annotations in annot files.')
    parser.add_argument('--ref-prefix',help='File prefix for chromosome seperated reference files.')
    parser.add_argument('--ref-suffix',help='File suffix for chromosome seperated reference files.')
    parser.add_argument('--file-prefix',help='File prefix for chromosome seperated files.')
    parser.add_argument('--file-suffix',help='File suffix for chromosome seperated files.')
    parser.add_argument('--output-prefix',help='Output file prefix for chromosome seperated files.')
    parser.add_argument('--output-suffix',help='Output file suffix for chromosome seperated files.')
    parser.add_argument('--input-file',help='Input file name.')
    parser.add_argument('--output-file',help='Output file name.')
    parser.add_argument('--threshold',type=float,help='Threshold for discretizing continuous variable.')
    args = parser.parse_args()

    if args.match_snps:
        match_snps(args.target_prefix,args.annot_prefix,args.annot_ref_prefix,args.target_output_prefix,args.annot_output_prefix,args.target_col_name,args.recompute)
    elif args.leave_one_out:
        leave_one_out(args.target_prefix,args.annot_prefix,args.target_output_prefix,args.annot_output_prefix,args.target_col_name,args.num_annots,args.recompute)
    elif args.match_ref:
        match_ref(args.ref_prefix,args.ref_suffix,args.file_prefix,args.file_suffix,args.output_prefix,args.output_suffix,args.recompute)
    elif args.discretize:
        discretize(args.input_file,args.target_col_name,args.threshold,args.output_file)
        if args.three_bins:
            discretize_threebins(args.input_file,args.target_col_name,args.output_file)
