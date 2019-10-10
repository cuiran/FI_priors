import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
import xgboost as xgb
import pdb
import argparse

def ordered_intersection(snp1,snp2):
    '''
    snp1 and snp2 are lists of SNPs with ID format CHR:BP:A1:A2
    '''
    l = list(set(snp1)&set(snp2))
    l.sort()
    return l

def read_annot(fname,snplist):
    '''
    snplist is a list of SNP ID's
    annotation corresponds to fname must contain all SNPs in snplist
    '''
    snp_df = pd.read_csv(fname,delim_whitespace=True,usecols=['SNP'])
    shared_snps_idx = snp_df[snp_df['SNP'].isin(snplist)].index.tolist()
    print('There are {} fine-mapped SNPs that are also annotated'.format(len(shared_snps_idx)))
    skip_idx = list(set(snp_df.index.tolist()).difference(set(shared_snps_idx)))
    skip_idx = [x+1 for x in skip_idx]
    cols = list(pd.read_csv(fname,delim_whitespace=True,nrows =1))
    annot_df = pd.read_csv(fname,
                       delim_whitespace=True,
                       usecols=[i for i in cols if i not in ['CHR','BP','CM']],
                       skiprows = skip_idx)
    annot_snps = annot_df['SNP'].tolist()
    annot_snps_sorted = sorted(annot_snps)
    if annot_snps_sorted!=snplist:
        raise ValueError('SNPs in filtered dataframe does not match ones given')
    annot_df.set_index('SNP',inplace=True)
    return annot_df

def read_annot_fm(fname,fm_df):
    '''
    fname is the file name for the annotation
    fm_snps is a list of snp names that are fine-mapped
    '''
    fm_snps = fm_df.index.tolist()
    snp_df = pd.read_csv(fname,delim_whitespace=True,usecols=['SNP'])
    shared_snps = ordered_intersection(fm_snps,snp_df.values.flatten())
    annot_df = read_annot(fname,shared_snps)
    annot_snps = annot_df.index.tolist()
    fm_filt = fm_df.loc[annot_snps,'prob']
    if fm_filt.index.tolist()!= annot_snps:
        raise ValueError('SNP order in annotation dataframe does not match fm dataframe')
    return annot_df,fm_filt.to_frame()

def binarize(df,cutoff):
    '''
    df has snp ID as index, only column contains prob
    '''
    ddf = df.copy()
    ddf['prob_binarized'] = np.where(df['prob']>=cutoff,1,0)
    new_df = ddf['prob_binarized'].to_frame()
    return new_df

def method_y_pred(method, X_train, y_train, X_test):
    print('Training {} predictor'.format(method))
    predictor = create_predictor(method)
    predictor.fit(X_train,y_train)
    print('Predicting with {} predictor'.format(method))
    y_pred = predict(method,predictor,X_test)
    return y_pred

def predict(method,predictor,X_test):
    if method in ['ols','lasso','elnet','gbt','rf']:
        y_pred = predictor.predict(X_test)
    return y_pred

def create_predictor(method):
    ols = linear_model.LinearRegression(fit_intercept=False)
    lasso = linear_model.LassoCV(cv=5,fit_intercept=False)
    elnet = linear_model.ElasticNetCV(l1_ratio = [.01, .1, .3, .5, .7, .9, .95, .99, 1],cv=5)
    gbt = ensemble.GradientBoostingRegressor(n_estimators=1000,max_depth=2)
    rf = ensemble.RandomForestRegressor(max_depth=2, random_state=0,n_estimators=1000) 
    
    if method=='ols':
        return ols
    elif method=='lasso':
        return lasso
    elif method=='elnet':
        return elnet
    elif method=='gbt':
        return gbt
    elif method=='rf':
        return rf

def score_function(y_pred,y_test):
    score = metrics.r2_score(y_test,y_pred)
    return score

def get_train_test(feature_dfs,target_dfs,skip_idx):
    X_train_df = pd.concat([feature_dfs[i] for i in range(len(feature_dfs)) if i!=skip_idx],axis=0)
    y_train_df = pd.concat([target_dfs[i] for i in range(len(target_dfs)) if i!=skip_idx],axis=0)
    X_test_df = feature_dfs[skip_idx]
    y_test_df = target_dfs[skip_idx]
    X_train = X_train_df.values
    y_train = y_train_df.values.reshape(-1)
    X_test = X_test_df.values
    y_test = y_test_df.values.reshape(-1)
    return X_train,y_train,X_test,y_test

def run_method(method,fm_fname,leave_chrom,annot_prefix,out):
    '''
    fm_fname is fine-mapping results file name
    annot_prefix can have multiple annotations, they need to be comma delimited.
    The first annot_prefix must be the baseline
    out is file prefix for output files, should include PHENO_ANNOTS_
    '''
    fm_df = pd.read_csv(fm_fname,delim_whitespace=True,dtype={'position':int,'chromosome':int})
    fm_df['v'] = fm_df['chromosome'].astype(str)+':'+fm_df['position'].astype(str)+':'+fm_df['allele1']+':'+fm_df['allele2']
    fm_df.set_index('v',inplace=True)
    annot_dfs,fm_filt_dfs = get_annots_from_files(annot_prefix,fm_df)
    print('Leaving chromosome {} out'.format(leave_chrom))
    skip_idx = int(leave_chrom)-1
    X_train,y_train,X_test,y_test = get_train_test(annot_dfs,fm_filt_dfs,skip_idx)
    y_pred = method_y_pred(method,X_train,y_train,X_test)
    pred_df = pd.DataFrame(None,columns=['YPRED','YTEST'])
    pred_df['YPRED'] = y_pred
    pred_df['YTEST'] = y_test
    pred_df.to_csv(out+method+'_leave'+leave_chrom+'.ypred',sep='\t',index=False)
    return

def get_annots_from_files(annot_prefix,fm_df):
    '''
    annot_prefix can have multiple annotations, they need to be comma delimited.
    The first annot_prefix must be the baseline

    return two lists; first is a list of annotation dataframes one for each chromosome
    second is a list of pip dataframes, one for each chromosome
    '''
    annot_df_list = [] # 22 lists of dfs
    fm_df_list = []
    annot_prefix_list = annot_prefix.split(',')
    for i in range(1,23):
        chrom = str(i)
        a_fname = a_prefix+chrom+'.annot.gz'
        chr_annot_df_list = []
        for a_prefix in annot_prefix_list:
            if a_prefix == annot_prefix_list[0]:
                a_df,f_df = read_annot_fm(a_fname,fm_df)
                snp_list = f_df.index.tolist()
                chr_annot_df_list.append(a_df)
                fm_df_list.append(f_df)
            else:
                a_df = read_annot(a_fname,snp_list)
                chr_annot_df_list.append(a_df)
        chr_annot_df = pd.concat(chr_annot_df_list,axis=1)
    annot_df_list.append(chr_annot_df)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',help='ols,lasso,elnet,gbt,rf')
    parser.add_argument('--fm-fname',help='fine-mapping result file name')
    parser.add_argument('--leave-chr',type=str,help='test chromosome')
    parser.add_argument('--annot-prefix',help='comma delimited prefixes')
    parser.add_argument('--out',help='output prefix of the form PHENO_ANNOT_')
    args = parser.parse_args()

    run_method(args.method,args.fm_fname,args.leave_chr,args.annot_prefix,args.out)

    
