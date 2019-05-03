import pandas as pd
import numpy as np
import argparse
import os
import sys
import fit
import pdb
import math
from sklearn import linear_model

'''
This script include the method that would fit some model using 20 chromosomes,
Using one of the other 2 chromosomes to bin target and the prediction of the 
one other left out chromosome would be the average of target in each bin.
'''

def run_pipeline(xtrain_prefix,ytrain_prefix,xtrain_suffix,ytrain_suffix,xtest_bin_prefix,xtest_bin_suffix,xtest_pred_prefix,xtest_pred_suffix,ytest_bin_prefix,ytest_bin_suffix,ytest_pred_prefix,ytest_pred_suffix,col_name,output_prefix,pred_chrom,bin_chrom,num_annots,method,num_bins,binning_method,recompute):
    '''
    xtrain_prefix is the prefix of chromosome seperated files that contains training regressors data.
    The files contain some information columns, standard columns are: 'SNP','BP','CHR','CM'. Following
    which are the training regressor data.

    ytrain_prefix is the prefix of chromosome seperated files that contains training target data.
    The files contian some information columns, standard columns are: 'CHR','BP','SNP','CM'. Following 
    which is the training target data. Typically the last column.

    xtest_prefix is the prefix of file which contains test regressor data.

    output_prefix is the prefix that you want your output to have. This should include phenotype information,
    method name and pred-chrom, bin-chrom information will be added in the function

    pred_chrom(str) is the chromosome on which you want the prediction of without looking at the true target. The prediction
    is based on aggregated true target values on bin_chrom. 

    bin_chrom(str) is the chromosome on which you bin predicted target and compute aggregated true target.

    num_annots is the number of annotations.
    ''' 
    fname_prefix_nobins = output_prefix+'_'+method+'_bin'+bin_chrom+'_pred'+pred_chrom+'_'+binning_method+'.'
    fname_prefix = output_prefix+'_'+method+'_'+str(num_bins)+'bins_bin'+bin_chrom+'_pred'+pred_chrom+'_'+binning_method+'.'
    ypred_fname = fname_prefix+'ybinpred'
    if os.path.isfile(ypred_fname) and not recompute:
        print("Predicted target file already exists at: "+ypred_fname)
        sys.exit()
    else:
        yrank_prefixes = [fname_prefix_nobins+bin_chrom+'.',fname_prefix_nobins+pred_chrom+'.']
        yrank_fnames = [f+'ypred' for f in yrank_prefixes]
        if not all([os.path.isfile(f) for f in yrank_fnames]):
            trainx,trainy = make_training_files(xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,output_prefix,pred_chrom,bin_chrom,num_annots,recompute)
            testx_bin = xtest_bin_prefix +bin_chrom+'.'+xtest_bin_suffix
            testx_pred = xtest_pred_prefix+pred_chrom+'.'+xtest_pred_suffix
            [ypred_binchrom,ypred_predchrom] = fit.fit_model_multipred(trainx,trainy,fname_prefix_nobins,[testx_bin,testx_pred],yrank_prefixes,method,num_annots)
        else:
            print('Ranking files already exist, reading them in...')
            ypred_binchrom = pd.read_csv(yrank_fnames[0],delim_whitespace=True).values
            ypred_predchrom = pd.read_csv(yrank_fnames[1],delim_whitespace=True).values
        bindf = pd.read_parquet(ytest_bin_prefix+bin_chrom+'.'+ytest_bin_suffix,engine='pyarrow')
        predf = pd.read_parquet(ytest_pred_prefix+pred_chrom+'.'+ytest_pred_suffix,engine='pyarrow')
        # max min scale the predictions into [0,1]
        ybin_stdized = min_max_scale(ypred_binchrom)
        ypred_stdized = min_max_scale(ypred_predchrom)
        bindf[bin_chrom] = ybin_stdized
        predf[pred_chrom] = ypred_stdized
        print('Getting predictions in each bin')
        bin_pred,cutoffs = get_bin_pred(bindf,bin_chrom,num_bins,col_name,binning_method)
        if any([math.isnan(x) for x in bin_pred]):
            print('Empty bins detected. Need smaller number of bins')
        else:
            binpreddf = pd.DataFrame(data=bin_pred,columns=['BIN_PRED'])
            binpreddf.to_csv(fname_prefix+'predinbins',sep='\t',index=False)
            cutoffdf = pd.DataFrame(data=cutoffs,columns=['CUTOFF'])
            cutoffdf.to_csv(fname_prefix+'cutoffs',sep='\t',index=False)
            print('Applying binned predictions on chromosome '+pred_chrom)
            predf[pred_chrom+'_binpred'] = predf.apply(lambda row:assign_binpred(row,pred_chrom,bin_pred,cutoffs),axis=1)
            predf.rename(columns={pred_chrom:'ypred_stdized',pred_chrom+'_binpred':'ybinpred'},inplace=True)
            predf['ypred_original'] = ypred_predchrom
            predf.to_csv(ypred_fname,sep='\t',index=False)
    return
      
def binpred_cv(xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,col_name,num_annots,num_bins,pred_chrom,output_prefix,recompute):
    '''
    The 'cv' in the name of this function is a misnomer, because we are not doing any kind of cross validation.
    However, the flavor of this method is kind of like the flavor of cross-validation, in the sense that we are picking 
    a different chromosome each time for binning, then average the resulting prediction on the chromosomes of interest at the end.

    The assumed model is linear regression with OLS. Binning method is to create equally sized bins (same number of SNPs).

    xtrain files has info columns: SNP, BP, CHR CM
    ytrain files has the same info columns

    col_name is the name of the target column, usually it's pip

    output_prefix should contain phenotype, annotation set, target type (PIP), binpred_cv method name. 
    '''
    fname = output_prefix+'_pred'+pred_chrom+'.'
    trainx_fname = fname+'trainx.tmp.parquet'
    trainy_fname = fname+'trainy.tmp.parquet'
    if not all([os.path.isfile(f) for f in [trainx_fname,trainy_fname]]):
        print('Concatenating training data')
        xdf,ydf = concat_training(trainx_fname,trainy_fname,xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,pred_chrom)
        xdf.to_parquet(trainx_fname,engine='pyarrow')
        ydf.to_parquet(trainy_fname,engine='pyarrow')
    else:
        print('Reading in training data')
        xdf = pd.read_parquet(trainx_fname,engine='pyarrow')
        ydf = pd.read_parquet(trainy_fname,engine='pyarrow')
    train_chroms = [x for x in range(1,23) if x!=int(pred_chrom)]
    binpreds = []
    for i in train_chroms:
        print('Computing binned predictions using binning chromosome '+str(i))
        ybinpred_fname = output_prefix+'_bin'+str(i)+'_pred'+str(pred_chrom)+'.'+str(num_bins)+'bins.ybinpred'
        if os.path.isfile(ybinpred_fname):
            ybinpreddf = pd.read_csv(ybinpred_fname,delim_whitespace=True)
            binned_prediction = ybinpreddf[pred_chrom+'_binpred'].values
        else:
            binned_prediction = binpred_single(xdf,ydf,i,pred_chrom,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,output_prefix,num_bins,num_annots,col_name)
        if binned_prediction is None:
            continue
        else:
            binpreds.append(binned_prediction)
    mean_binpred = np.mean(binpreds,axis=0)
    if ytest_suffix == 'parquet':
        preddf = pd.read_parquet(ytest_prefix+str(pred_chrom)+'.'+ytest_suffix,engine='pyarrow')
    else:
        preddf = pd.read_csv(ytest_prefix+str(pred_chrom)+'.'+ytest_suffix,delim_whitespace=True)
    preddf['mean_binpred'] = mean_binpred
    preddf.to_csv(fname+str(num_bins)+'_bins.ybinpred',sep='\t',index=False)
    return

def binpred_eo(xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,col_name,num_annots,num_bins,pred_chrom,output_prefix,train_eo,recompute):
    '''
    "eo" means even-odd. This function implement the method of binned prediction where we split the training/binning data into even and odd chromosomes.
    
    The assumed model is linear regression with OLS. Binning method is to create equally sized bins (same number of SNPs).

    xtrain files has info columns: SNP, BP, CHR CM
    ytrain files has the same info columns
    
    xtrain files should contain all SNPs except for those on the test chromosome
    ytrain files should contain all SNPs except for those on the test chromosome

    xtrain prefix should stop right before the chromosome number
    xtrain suffix should contain all string after chromosome number and '.'

    col_name is the name of the target column, usually it's pip

    train_eo takes argument "even" or "odd" indicating which set of SNPs are used for trainning and the other set of SNPs are for binning.

    output_prefix should contain phenotype, annotation set, target type (PIP), binpred_eo method name.
    '''
    print('Reading in training data')
    xdf = pd.read_parquet(xtrain_prefix+pred_chrom+'.'+xtrain_suffix,engine='pyarrow')
    ydf = pd.read_parquet(ytrain_prefix+pred_chrom+'.'+ytrain_suffix,engine='pyarrow')
    train_chroms,bin_chroms = get_train_bin_chroms(pred_chrom,train_eo)
    binned_prediction = binpred_single(xdf,ydf,bin_chroms,pred_chrom,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,output_prefix,num_bins,num_annots,col_name)
    if binned_prediction is None:
        print('Failed to compute binned prediction. Likely due to no SNPs in binning chromosomes')
    else:
        if ytest_suffix == 'parquet':
            preddf = pd.read_parquet(ytest_prefix+str(pred_chrom)+'.'+ytest_suffix,engine='pyarrow')
        preddf['binpred'] = binned_prediction
        if train_eo == 'even':
            outfname = output_prefix+'_binodd_pred'+str(pred_chrom)+'.'+str(num_bins)+'bins.ybinpred'
        elif train_eo == 'odd':
            outfname = output_prefix+'_bineven_pred'+str(pred_chrom)+'.'+str(num_bins)+'bins.ybinpred'
        preddf.to_csv(outfname,sep='\t',index=False)
    return


def get_train_bin_chroms(pred_chrom,train_eo):
    '''
    This function will output two arrays, the first one is the list of chromosome numbers for training
    the second list is the list of chromosome numbers for binning

    train_eo is either "even" or "odd"

    if train_eo is "even" then bin_eo is "odd"
    '''
    chroms = [x for x in range(1,23) if x!=int(pred_chrom)]
    even_chroms = [i for i in chroms if i%2==0]
    odd_chroms = [i for i in chroms if i%2 != 0]
    if train_eo == "even":
        return even_chroms, odd_chroms
    elif train_eo == "odd":
        return odd_chroms,even_chroms
    else:
        print('Invalid value for train_eo')
        return
    

def binpred_single(xdf,ydf,binchrom,predchrom,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,output_prefix,num_bins,num_annots,col_name):
    '''
    This function computes the binned prediction on the chromosome of interest

    xdf (ydf) is a dataframe of training predictors (target) with info columns: SNP, BP, CHR, CM

    binchrom is a list of chromosomes for binning 
    predchrom is the chromosome of interest

    col_name is usually pip
    '''
    if len(binchrom)>1:
        if binchrom[0]%2==0:
            fname_nobin = output_prefix+'_bineven_pred'+str(predchrom)+'.'
        else:
            fname_nobin = output_prefix+'_binodd_pred'+str(predchrom)+'.'
    else:
        fname_nobin = output_prefix+'_bin'+str(binchrom)+'_pred'+str(predchrom)+'.'
    if os.path.isfile(fname_nobin+'coef'):
        fitted_coefs = pd.read_csv(fname_nobin+'coef',delim_whitespace=True).values
    else:
        fitted_coefs = fit_ols(xdf,ydf,binchrom,num_annots,col_name)
        coefdf = pd.DataFrame(data=fitted_coefs,columns=['COEF'])
        coefdf.to_csv(fname_nobin+'coef',sep='\t',index=False)
    ypred_fnames = [fname_nobin+'binchrom.ypred',fname_nobin+str(predchrom)+'.ypred']
    if not all([os.path.isfile(f) for f in ypred_fnames]):
        yhat_bindf,yhat_preddf = ols_pred(fitted_coefs,xdf,ydf,binchrom,predchrom,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,num_annots,ypred_fnames)
    else:
        yhat_bindf,yhat_preddf = [pd.read_csv(f,delim_whitespace=True) for f in ypred_fnames]
    if yhat_bindf is None:
        return None
    else:
        scaledyhat_bin = min_max_scale(yhat_bindf['YPRED'].values)
        if len(binchrom)>1:
            yhat_bindf['scaledyhat_bin'] = scaledyhat_bin
        else:
            yhat_bindf[binchrom] = scaledyhat_bin
        scaledyhat_pred = min_max_scale(yhat_preddf['YPRED'].values)
        yhat_preddf[predchrom] = scaledyhat_pred
        print('Getting predictions in each bin')
        bin_pred,cutoffs = get_bin_pred(yhat_bindf,binchrom,num_bins,col_name,'equally-sized')
        if any([math.isnan(x) for x in bin_pred]):
            print('Empty bins detected. Need smaller number of bins')
        else:
            fname = fname_nobin+str(num_bins)+'bins.'
            binpreddf = pd.DataFrame(data=bin_pred,columns=['BIN_PRED'])
            binpreddf.to_csv(fname+'predinbins',sep='\t',index=False)
            cutoffdf = pd.DataFrame(data=cutoffs,columns=['CUTOFF'])
            cutoffdf.to_csv(fname+'cutoffs',sep='\t',index=False)
            print('Applying binned predictions on chromosome '+str(predchrom))
            yhat_preddf[predchrom+'_binpred'] = yhat_preddf.apply(lambda row:assign_binpred(row,predchrom,bin_pred,cutoffs),axis=1)
            yhat_preddf.to_csv(fname+'ybinpred',sep='\t',index=False)
        return yhat_preddf[predchrom+'_binpred'].values
        


def ols_pred(fitted_coefs,xdf,ydf,binchrom,predchrom,xtest_prefix,xtest_suffix,ytest_prefix,ytest_suffix,num_annots,ypred_fnames):
    xbindf = xdf[xdf['CHR'].isin(binchrom)]
    if xbindf.empty:
        return [None,None]
    else:
        if xtest_suffix=='parquet':
            xpreddf = pd.read_parquet(xtest_prefix+predchrom+'.'+xtest_suffix,engine='pyarrow')
        xtests = [xbindf,xpreddf]
        ypreds = []
        for i in range(2):
            ypred_fname = ypred_fnames[i]
            if os.path.isfile(ypred_fname):
                ypred = pd.read_csv(ypred_fname,delim_whitespace=True)
            else:
                num_annots = int(num_annots)
                ypred_array = xtests[i].iloc[:,-num_annots:].values.dot(fitted_coefs)
                if i == 0:
                    ypred = ydf[ydf['CHR'].isin(binchrom)].copy()
                else:
                    ypred = pd.read_parquet(ytest_prefix+str(predchrom)+'.'+ytest_suffix,engine='pyarrow')
                ypred['YPRED'] = ypred_array
            ypred.to_csv(ypred_fname,sep='\t',index=False)
            ypreds.append(ypred)
        return ypreds

def fit_ols(xdf,ydf,exclude_chrom,num_annots,col_name):
    '''
    Fit OLS with xdf as predictors and ydf as target

    The last num_annots columns of xdf are the regression data

    The column with col_name in ydf is the target data

    Fitting should exclude the chromosomes specified by the list exclude_chrom
    '''
    print('Fitting model leaving out binning chromosome '+str(exclude_chrom))
    x,y = get_xy(xdf,ydf,exclude_chrom,num_annots,col_name)
    fitted_coefs = ols(x,y)
    return fitted_coefs

def get_xy(xdf,ydf,exclude_chrom,num_annots,col_name):
    print('Getting the training data leaving out binning chromosome '+str(exclude_chrom))
    num_annots = int(num_annots)
    if len(exclude_chrom) > 1:
        trainx = xdf[~xdf['CHR'].isin(exclude_chrom)].iloc[:,-num_annots:].values
        trainy = ydf[~ydf['CHR'].isin(exclude_chrom)][col_name].values
    else:
        trainx = xdf[xdf['CHR']!=exclude_chrom].iloc[:,-num_annots:].values
        trainy = ydf[ydf['CHR']!=exclude_chrom][col_name].values
    return trainx,trainy

def ols(x,y):
    print('Performing OLS regression')
    ols_model = linear_model.LinearRegression(fit_intercept=False)
    ols_model.fit(x,y)
    return ols_model.coef_.ravel()

def concat_training(trainx_fname,trainy_fname,xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,pred_chrom):
    '''
    Concatenate training files
    
    trainx_fname is the output file name of training predictors
    trainy_fname is the output file name of training targets

    xtrain_prefix and xtrain_suffix are the prefix and suffix of the chromosome separated training file names

    This function returns two dataframes: xdf and ydf with the info columns SNP, BP, CHR, CM
    '''
    if os.path.isfile(trainx_fname) and not recompute:
        print('Concatenated training predictor file already exists at: '+trainx_fname)
        xdf = pd.read_parquet(trainx_fname)
    else:
        toconcat = [xtrain_prefix+str(i)+'.parquet' for i in range(1,23) if i!=int(pred_chrom)]
        xdf = concat_dfs(toconcat)
    if os.path.isfile(trainy_fname) and not recompute:
        print('Concatenated training target file already exists at:'+trainy_fname)
        ydf = pd.read_parquet(trainy_fname)
    else:
        toconcat = [ytrain_prefix+str(i)+'.parquet' for i in range(1,23) if i!=int(pred_chrom)]
        ydf = concat_dfs(toconcat)
    return xdf,ydf
    
 
def get_bin_pred(bindf,bin_chrom,num_bins,col_name,binning_method):
    '''
    Generate probability predictions for each bin and the bin cutoffs according to the binning method
    
    bindf is a dataframe with columns bin_chrom to store predictions for the binning chromosome, 
    col_name stores true probabilities for the binning chromosome

    binning_method is either "equally-spaced" or "equally-sized"

    return:
    bin_pred: list of length num_bins with the proper prediction for each bin from small to large
    cutoffs: the bin cutoffs from small to large, always start with 0.0 and ends with 1.0
    '''
    if len(bin_chrom)>1:
        bin_chrom = 'scaledyhat_bin'
    if binning_method == 'equally-spaced':
        cutoffs = [float(i)/num_bins for i in range(num_bins+1)]
        bin_pred = [bindf[(bindf[bin_chrom]>=cutoffs[i])&(bindf[bin_chrom]<cutoffs[i+1])].ix[:,col_name].mean() for i in range(num_bins)]+[bindf[(bindf[bin_chrom]>=cutoffs[-2])&(bindf[bin_chrom]<=cutoffs[-1])].ix[:,col_name].mean()] 
    elif binning_method == 'equally-sized':
        sorted_df = bindf.sort_values(by=[bin_chrom])
        num_snps_perbin = sorted_df.shape[0]//num_bins
        cutoffs= [0.0]+[sorted_df.iloc[num_snps_perbin*i-1,:][bin_chrom] for i in range(1,num_bins)]+[1.0]
        bin_pred = [sorted_df.iloc[num_snps_perbin*i:num_snps_perbin*(i+1),:][col_name].mean() for i in range(num_bins-1)]+[sorted_df.iloc[num_snps_perbin*(num_bins-1):,:][col_name].mean()]
    return bin_pred,cutoffs

def min_max_scale(a):
    '''
    Scaling array a to range [0,1] using the fomula below
    (a-min(a))/(max(a)-min(a))
    '''
    mi = float(min(a))
    ma = float(max(a))
    diff = ma-mi
    return (a-mi)/diff

def assign_binpred(row,pred_chrom,bin_pred,cutoffs):
    num_bins = len(bin_pred)
    for i in range(num_bins-1):
        if (row[pred_chrom]>=cutoffs[i]) and (row[pred_chrom]<cutoffs[i+1]):
            return bin_pred[i]
    if (row[pred_chrom]>=cutoffs[-2]) and (row[pred_chrom]<=cutoffs[-1]):
        return bin_pred[-1]

def make_training_files(xtrain_prefix,xtrain_suffix,ytrain_prefix,ytrain_suffix,output_prefix,pred_chrom,bin_chrom,num_annots,recompute):
    '''
    This function outputs file names for the training data.
    The output files should contain no information columns, just the data.
    '''
    trainx_fname = output_prefix+'_bin'+bin_chrom+'_pred'+pred_chrom+'_trainx.tmp.parquet'
    trainy_fname = output_prefix+'_bin'+bin_chrom+'_pred'+pred_chrom+'_trainy.tmp.parquet'
    train_chroms = range(1,23)
    train_chroms.remove(int(pred_chrom))
    train_chroms.remove(int(bin_chrom))
    if os.path.isfile(trainx_fname) and not recompute:
        print('File exists at '+trainx_fname)
    else:
        print('Gathering training predictor data')
        xfiles = [xtrain_prefix+str(i)+'.'+xtrain_suffix for i in train_chroms]
        xdf = concat_dfs(xfiles)
        xdf_data = xdf.iloc[:,-int(num_annots):]
        print('Training predictor data shape is '+str(xdf_data.shape))
        print('Saving training predictor data to '+trainx_fname)
        xdf_data.to_parquet(trainx_fname,engine='pyarrow')
    if os.path.isfile(trainy_fname) and not recompute:
        print('File exists at '+trainy_fname)
    else:
        print('Gathering training target data')
        yfiles = [ytrain_prefix+str(i)+'.'+ytrain_suffix for i in train_chroms]
        ydf = concat_dfs(yfiles)
        ydf_data = ydf.iloc[:,-1].to_frame()
        print('Training target data shape is '+str(ydf_data.shape))
        print('Saving training target data to '+trainy_fname)
        ydf_data.to_parquet(trainy_fname,engine='pyarrow')
    return trainx_fname,trainy_fname


def concat_dfs(fname_list):
    if 'parquet' in fname_list[0]:
        dfs = [pd.read_parquet(f,engine='pyarrow') for f in fname_list]
    else:
        dfs = [pd.read_csv(f,delim_whitespace=True) for f in fname_list]
    df = pd.concat(dfs,axis=0,sort=False)
    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xtrain-prefix',help='File prefix for training predictor files, prefix should include everything right before the chromosome number.')
    parser.add_argument('--xtrain-suffix',help='File suffix for training predictor files, suffix should include everything after chrom number except for the dot')
    parser.add_argument('--ytrain-prefix',help='See help for xtrain-prefix')
    parser.add_argument('--ytrain-suffix',help='See help for xtrain-suffix')
    parser.add_argument('--xtest-prefix',help='See help for xtrain-prefix')
    parser.add_argument('--xtest-suffix',help='See help for xtrain-suffix')
    parser.add_argument('--ytest-prefix',help='See help for xtrain-prefix')
    parser.add_argument('--ytest-suffix',help='See help for xtrain-suffix')
    parser.add_argument('--xtest-bin-prefix')
    parser.add_argument('--xtest-bin-suffix')
    parser.add_argument('--xtest-pred-prefix')
    parser.add_argument('--xtest-pred-suffix')
    parser.add_argument('--ytest-bin-prefix')
    parser.add_argument('--ytest-bin-suffix')
    parser.add_argument('--ytest-pred-prefix')
    parser.add_argument('--ytest-pred-suffix')
    parser.add_argument('--output-prefix',help='This should include phenotype information. Bin chrom number and predict chrom number will be added in function')
    parser.add_argument('--pred-chrom',help='Chromosome on which we want prediction.')
    parser.add_argument('--bin-chrom',help='Chromosome number on which we create bins and final predictions for those bins.')
    parser.add_argument('--num-annots',help='Number of annotations in file. If there are multiple files, use comma as delimiter')
    parser.add_argument('--method',help='Fitting method: OLS, Lasso, Tree, GBT, RF, Logit, Logistic')
    parser.add_argument('--col-name',help='Column name of target values')
    parser.add_argument('--num-bins',type=int,help='Number of bins')
    parser.add_argument('--binning-method',help='Currently taking values equally-spaced or equally-sized')
    parser.add_argument('--recompute',action='store_true')
    parser.add_argument('--mean-binpred',action='store_true')
    parser.add_argument('--single-binchrom',action='store_true',help='Oldest method, binning only using one specified chromosome')
    parser.add_argument('--train-eo',help='Specify even or odd chromosome for training')
    args = parser.parse_args()

    if args.single_binchrom:
        run_pipeline(args.xtrain_prefix,args.ytrain_prefix,args.xtrain_suffix,args.ytrain_suffix,args.xtest_bin_prefix,args.xtest_bin_suffix,args.xtest_pred_prefix,args.xtest_pred_suffix,args.ytest_bin_prefix,args.ytest_bin_suffix,args.ytest_pred_prefix,args.ytest_pred_suffix,args.col_name,args.output_prefix,args.pred_chrom,args.bin_chrom,args.num_annots,args.method,args.num_bins,args.binning_method,args.recompute)
    elif args.mean_binpred:
        binpred_cv(args.xtrain_prefix,args.xtrain_suffix,args.ytrain_prefix,args.ytrain_suffix,args.xtest_prefix,args.xtest_suffix,args.ytest_prefix,args.ytest_suffix,args.col_name,args.num_annots,args.num_bins,args.pred_chrom,args.output_prefix,args.recompute)
    elif args.train_eo:
        binpred_eo(args.xtrain_prefix,args.xtrain_suffix,args.ytrain_prefix,args.ytrain_suffix,args.xtest_prefix,args.xtest_suffix,args.ytest_prefix,args.ytest_suffix,args.col_name,args.num_annots,args.num_bins,args.pred_chrom,args.output_prefix,args.train_eo,args.recompute)
