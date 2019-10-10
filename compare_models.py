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

def method_r2_score(method, X_train, y_train, X_test, y_test):
    print('Training {} predictor'.format(method))
    predictor = create_predictor(method)
    predictor.fit(X_train,y_train)
    print('Predicting with {} predictor'.format(method))
    y_pred = predict(method,predictor,X_test)
    score = score_function(y_pred,y_test)
    return score

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

def score_all_methods(fm_fname,leave_chrom,annot_prefix,out):
    '''
    fm_fname is fine-mapping results file name
    annot_prefix can have multiple annotations, they need to be comma delimited.
    The first annot_prefix must be the baseline
    '''
    fm_df = pd.read_csv(fm_fname,delim_whitespace=True,dtype={'position':int,'chromosome':int})
    fm_df['v'] = fm_df['chromosome'].astype(str)+':'+fm_df['position'].astype(str)+':'+fm_df['allele1']+':'+fm_df['allele2']
    fm_df.set_index('v',inplace=True)
    annot_dfs,fm_filt_dfs = get_annots_from_files(annot_prefix,fm_df)
    print('Leaving chromosome {} out'.format(leave_chrom))
    skip_idx = int(leave_chrom)-1
    X_train,y_train,X_test,y_test = get_train_test(annot_dfs,fm_filt_dfs,skip_idx)
    ols_score = method_r2_score('ols', X_train, y_train, X_test, y_test))
    lasso_score = method_r2_score('lasso', X_train, y_train, X_test, y_test)
    elnet_score = method_r2_score('elnet', X_train, y_train, X_test, y_test)
    gbt_score = method_r2_score('gbt', X_train, y_train, X_test, y_test)
    rf_score = method_r2_score('rf', X_train, y_train, X_test, y_test)
    df = pd.DataFrame(None,columnns=['Method','R2_score'])
    df['Method'] = ['ols','lasso','elnet','gbt','rf']
    df['R2_score'] = [ols_score,lasso_score,elnet_score,gbt_score,rf_score]
    df.to_csv(out+'.leave_'+leave_chrom+'.r2_score',sep='\t',index=False)
    return
