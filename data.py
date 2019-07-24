import pandas as pd
import numpy as np
import argparse
import sys
import glob
import pdb
from utils import memoize
import utils as u

class Feature:

    def __init__(self,ref_annot_prefix=None,cts_annot_prefix=None,thin=False,
                 cts_result_folder=None,ldcts_folder=None,cts_p_cutoff=0.1,
                 info_colnames=['CHR','SNP','CM','BP']):
        self.ref_annot_prefix = ref_annot_prefix
        self.cts_annot_prefix = cts_annot_prefix
        self.cts_result_folder = cts_result_folder # folder for cell_type_result.txt files
        self.ldcts_folder = ldcts_folder # folder for ldcts files
        self.info_colnames = info_colnames

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
            if not u.check_shape(ref_df,cts_df,'row'):
                raise Exception('Shape of reference annotations doesn not match cell type specific annotations.\n
                                Reference annotation shape {},\n
                                cell type specific annotation shape {}'.format(ref_df.shape,cts_df.shape))
            else:
                if u.containList(self.info_colnames,cts_df.columns.tolist()):
                    cts_df.drop(self.info_colnames,axis=1,inplace=True)
                    return pd.concat([ref_df,cts_df],axis=1)
                elif u.nonInList(self.info_colnames,cts_df.columns.tolist()):
                    return pd.concat([ref_df,cts_df],axis=1)
                else:
                    raise Exception('Cell type specific annotations dataframe contains 
                                    some but not all info column names')

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
            cts_df = get_cts_annot_df(self.cts_result_folder,self.ldcts_folder,
                                      self.cts_p_cutoff,self.info_colnames,chrom)
            cts_df.to_csv(Prefix('cts_annot','out'))
            return cts_df
        else:
            chrom = str(chrom)
            return pd.read_csv(self.cts_annot_prefix+chrom+'.annot.gz',delim_whitespace=True)


def get_cts_annot_df(cts_folder,ldcts_folder,cts_p,info_colnames,chrom):
    '''This function uses information in all the cell_type_results.txt files 
    and the corresponding ldcts files to combine the cell type specific annotations
    into one dataframe. The p cutoff determines which cell types are significant enough
    to be included in this annotation dataframe.

    Each cell_type_result file is named as DATASET_REFANNOT_REFGENO.cell_type_results.txt, 
    REFGENO may have underscore like '1000G_EUR_Phase3', DATASET and REFANNOT shouldn't
    have underscore.
    These files have 4 columns: ['Name','Coefficient','Coefficient_std_error','Coefficient_P_value']

    The .ldcts files has two columns (no headers) first is the annotation name,
    second column is comma dilimited prefixes. Could be either just the annotation prefix 
    or the annotation prefix ',' control annotation prefix
    If there's control annotations, then 
    '''
    cts_annot_prefixes = get_cts_annot_prefixes(cts_folder,ldcts_folder,cts_p)
    fnames = [x+chrom+'.annot.gz' for x in cts_annot_prefixes['Prefix'].tolist()]
    dfs = [pd.read_csv(f,delim_whitespace=True) for f in fnames]
    info_cols,annot_dfs = sep_annot_df(dfs,info_colnames)
    for i in range(annot_dfs.shape[0]):
        df = annot_dfs[i]
        annot_name = cts_annot_prefixes.loc[i,'Name']
        if df.shape[1] == 1:
            # df has 1 annotation
            df.columns = [annot_name]
        elif df.shape[1] > 1:
            df.columns = [annot_name+'_'+x for x in df.columns.tolist()]
        else:
            raise Exception('Cell type specific annotation dataframe is empty')
    cts_annot_df = pd.concat([info_cols]+annot_dfs,axis=1)
    return cts_annot_df


def sep_annot_df(a,info_colnames):
    '''a can either be a list of dataframes or
    just one dataframe. 
    The 2 outputs are:
    info_columns: either empty or the columns that are in info_colnames
    dataframe(s): without the info_columns
    '''
    if type(a)==list:
        contain_info = [u.containList(info_colnames,x.columns.tolist()) for x in a]
        if any(contain_info):
            df_withinfo = a[contain_info.index(True)]
            info_df = df_withinfo[info_colnames].copy()
        else:
            info_df = pd.DataFrame(None)
        a_new = [u.dropCols(x,info_colnames) for x in a]
    else:
        if u.containList(info_colnames,a.columns.tolist()):
            info_df = a[info_columns].copy()
            a_new = a.drop(info_columns,axis=1)
    return info_df,a_new


def get_cts_annot_prefixes(cts_folder,ldcts_folder,cts_p):
    '''Generate a dataframe with two columns: Name and Prefix
    Name is the name of the annotation
    Prefix is the prefix for the cell type specific annotations
    we will use in the analysis. This should include any control annotations.

    Each cell_type_result file is named as DATASET_REFANNOT_REFGENO.cell_type_results.txt,
    REFGENO may have underscore like '1000G_EUR_Phase3', DATASET and REFANNOT shouldn't
    have underscore.
    These files have 4 columns: ['Name','Coefficient','Coefficient_std_error','Coefficient_P_value'] 
    '''
    # get cell type specific annotation names according to the pval cutoff
    cts_result_fnames = glob.glob(cts_folder+'*.cell_type_results.txt')
    cts_annot_names = dict()
    for f in cts_result_fnames:
        dataset_name = u.strip_dir(f).split('_')[0]
        df = pd.read_csv(f,delim_whitespace=True)
        cts_annot_names[dataset_name] = df[df['Coefficient_P_value']<=cts_p]['Name'].values.tolist()
    # get file prefixes based on the cts_annot_names
    ldcts_fnames = glob.glob(ldcts_folder+'*.ldcts')
    prefix_dfs = []
    for dataset_name in cts_annot_names:
        if len(cts_annot_names[dataset_name])>0:
            ldcts_fnames = glob.glob(ldcts_folder+'*'+dataset_name+'*.ldcts')
            if len(ldcts_fnames)>1:
                raise Exception('There are more than one ldcts file 
                                for dataset {} in folder {}'.format(dataset_name,ldcts_folder))
            else:
                ldcts_fname = ldcts_fnames[0]
            ldcts_df = pd.read_csv(ldcts_fname,delim_whitespace=True,header=None)
            prefix_df = ldcts_df[ldcts_df[0].isin(cts_annot_names[dataset_name])]
            prefix_df.columns=['Name','Prefix']
            if ',' in prefix_df.loc[0,'Prefix']:
                # if there's a comma in the prefix, then there's control annotations
                control_df = pd.DataFrame(None,columns=['Name','Prefix'])
                control_df['Name'] = dataset_name+'_control'
                control_df['Prefix'] = prefix_df.loc[0,'Prefix'].split(',')[1]
                prefix_df['Prefix'] = [x.split(',')[0] for x in prefix_df['Prefix'].tolist()]
                prefix_df = pd.concat([prefix_df,control_df],axis=0)
                prefix_dfs.append(prefix_df)
        else:
            continue
    return pd.concat(prefix_dfs,axis=0)


class Target:

    def __init__(self,target_file,target_colname='prob'):
        self.target_file = target_file # usually PIP file with all fine-mapped SNPs
        self.target_colname = target_colname # usually the column name for PIP



def Prefix(name,in_out):
    prefix_dict = {'feature_out':args.feature_output_prefix,
                   'cts_annot_in':'placeholder',
                   'cts_annot_out':'placeholder',
                   'ref_annot_in':args.ref_annot_prefix
                  }
    return prefix_dict[name+'_'+in_out]
