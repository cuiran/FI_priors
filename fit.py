import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
import argparse
import pdb
import os
import sys
import scipy

def get_xy(xfile,yfile): 
    pdb.set_trace()
    ydf = pd.read_parquet(yfile,engine='pyarrow')
    if ',' not in xfile:
        xdf = pd.read_parquet(xfile,engine='pyarrow')
    else:
        xfnames = xfile.split(',')
        xdfs = [pd.read_parquet(f,engine='pyarrow') for f in xfnames]
        xdf = pd.concat(xdfs,axis=1)
    y = ydf.values
    x = xdf.values
    return x,y.ravel()

def get_testx(test_fname,num_annots):
    if ',' not in test_fname:
        if 'parquet' in test_fname:
            xtestdf = pd.read_parquet(test_fname,engine='pyarrow')
        else:
            xtestdf = pd.read_csv(test_fname,delim_whitespace=True)
        xtest = xtestdf.iloc[:,-int(num_annots):].values
    else:
        fnames = test_fname.split(',')
        nums = num_annots.split(',')
        nums = [int(i) for i in nums]
        xdfs = [pd.read_parquet(a[0],engine='pyarrow').iloc[:,-a[1]:] for a in zip(fnames,nums)]
        xdf = pd.concat(xdfs,axis=1)
        xtest = xdf.values
    return xtest

def ols(x,y):
    print("Performing OLS regression")
    ols_model = linear_model.LinearRegression(fit_intercept=False)
    ols_model.fit(x,y)
    return ols_model.coef_.ravel()

def lasso(x,y):
    print("Performing Lasso regression")
    lasso_model = linear_model.LassoCV(fit_intercept=False)
    lasso_model.fit(x,y)
    return lasso_model.coef_.ravel(),lasso_model.alpha_

def gbt(x,y):
    print("Constructing gradient boosted trees")
    gbt_model = ensemble.GradientBoostingRegressor(n_estimators=100,max_depth=2)
    est = gbt_model.fit(x,y)
    return est

def decision_tree(x,y):
    print("Constructing decision tree regressor")
    tree_model = tree.DecisionTreeRegressor()
    est = tree_model.fit(x,y)
    return est

def rf(x,y):
    print("Constructing random forest")
    rf_model = ensemble.RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    est = rf_model.fit(x,y)
    return est

def logit(x,y):
    print("Performing logit regression")
    print("Transforming target using logit function")
    y[y==0] = np.min(y[y!=0])
    y[y==1] = np.max(y[y!=1])
    y_new = scipy.special.logit(y)
    fitted_coef = ols(x,y_new)
    return fitted_coef

def logistic(x,y):
    print("Fitting logistic regression.")
    logistic_model = linear_model.LogisticRegression(class_weight='balanced')
    est = logistic_model.fit(x,y)
    return est

def logistic_multi(x,y):
    print("Fitting logistic regression for multi class.")
    logistic_model = linear_model.LogisticRegression(class_weight='balanced',multi_class='multinomial',solver='saga')
    est = logistic_model.fit(x,y)
    return est
    

def fit_main(trainx,trainy,testx,output_prefix,leave_out,method,num_annots,recompute):
    '''
    Run regression with features in trainx, and target in trainy. 
    Assuming trainx and trainy contain only data and nothing else.
    testx and testy contains the testing data.

    This function will write out coefficients, regularization parameter
    predicted y, and mean squared error for appropriate methods.

    leave_out specifies which chromosome the regression will leave out.

    method specifies which method is used: OLS, Lasso, etc.
    '''
    fname_prefix = output_prefix+'_NOT'+leave_out+'_'+method+'.'
    ypred_fname = fname_prefix+'ypred'
    if os.path.isfile(ypred_fname) and not recompute:
        print("Predicted target file already exists")
        sys.exit()
    else:
        print('Fitting model')
        fit_model(trainx,trainy,testx,fname_prefix,method,num_annots)
    return

def fit_model_multipred(trainx,trainy,fname_train_prefix,testx_multi,fname_pred_prefix,method,num_annots):
    '''
    Run model fitting on a set of training data using method specified.

    This function can do prediction on multiple test data.

    testx is a list of test data.
    fname_train_prefix is the prefix of files that are generated without the chromosome of interest.
    fname_pred_prefix is a list of different names for the prediction files on different test data.
    
    The order of data in textx must match the order of the file prefixes in fname_pred_prefix.
    '''
    if not os.path.isfile(fname_train_prefix+'coef'):
        print('Loading in training data...')
        x,y = get_xy(trainx,trainy)
        if method == 'OLS':
            fitted_coefs = ols(x,y)
            coefdf = pd.DataFrame(data=fitted_coefs,columns=['COEF'])
            coefdf.to_csv(fname_train_prefix+'coef',sep='\t',index=False)
    else:
        fitted_coefs = pd.read_csv(fname_train_prefix+'coef',delim_whitespace=True).values
    if len(testx_multi)!=len(fname_pred_prefix):
        print('Length of text data doesn not match the length of output file names given.')
    else:
        m = len(testx_multi)
        pred_multi = list()
        for i in range(m):
            testx = testx_multi[i]
            fname_prefix = fname_pred_prefix[i]
            if not os.path.isfile(fname_prefix+'ypred'):
                xtest = get_testx(testx,num_annots)
                if method=='OLS':
                    ypred = xtest.dot(fitted_coefs)
                ypreddf = pd.DataFrame(data=ypred,columns=['YPRED'])
                ypreddf.to_csv(fname_prefix+'ypred',sep='\t',index=False)
            else:
                ypred = pd.read_csv(fname_prefix+'ypred',delim_whitespace=True).values
            pred_multi.append(ypred)
    return pred_multi



def fit_model(trainx,trainy,testx,fname_prefix,method,num_annots):
    '''
    Run model fitting, depending on the method.
    trainx, trainy should be file names of regressors and regression target.
    The files should have nothing else except for data.

    testx is file name for the test regressor data, it can have other info than the data itself.

    fname_prefix already include information about left-out chromosome and method
    '''
    if not os.path.isfile(fname_prefix+'coef'):
        print('Loading in training data...')
        x,y = get_xy(trainx,trainy)
    if method=='OLS':
        if not os.path.isfile(fname_prefix+'coef'):
            fitted_coefs = ols(x,y)
    elif method=='Lasso':
        if not os.path.isfile(fname_prefix+'coef'):
            fitted_coefs,reg_param = lasso(x,y)
            regparamdf = pd.DataFrame(data=[reg_param],columns=['ALPHA'])
            regparamdf.to_csv(fname_prefix+'alpha',sep='\t',index=False)
    elif method == 'Tree':
        est = decision_tree(x,y)
    elif method=='GBT':
        est = gbt(x,y)
    elif method == 'RF':
        est = rf(x,y)
    elif method == 'Logit':
        fitted_coefs = logit(x,y)
    elif method == 'Logistic':
        est = logistic(x,y)
    elif method == 'Logistic_multi':
        est = logistic_multi(x,y)
    xtest = get_testx(testx,num_annots)
    
    if method in ['OLS','Lasso','Logit']:
        if os.path.isfile(fname_prefix+'coef'):
            fitted_coefs = pd.read_csv(fname_prefix+'coef',delim_whitespace=True).values
        else:
            coefdf = pd.DataFrame(data=fitted_coefs,columns=['COEF'])
            coefdf.to_csv(fname_prefix+'coef',sep='\t',index=False)
        if method in ['Logit']:
            ypred = scipy.special.expit(xtest.dot(fitted_coefs))
        elif method in ['OLS','Lasso']:
            ypred = xtest.dot(fitted_coefs) 
    elif method in ['GBT','RF','Tree']:
        ypred = est.predict(xtest)
        if method == 'Tree':
            imp = est.feature_importances_
            impdf = pd.DataFrame(data=imp,columns=['IMPORTANCE'])
            impdf.to_csv(fname_prefix+'feature_importances',sep='\t',index=False)
    elif method in ['Logistic']:
        ypred = est.predict_proba(xtest)[:,1]
    elif method in ['Logistic_multi']:
        ypred = est.predict(xtest)
        ypredprob = est.predict_proba(xtest)
        yprobdf = pd.DataFrame(data=ypredprob,columns=['0','1','2'])
        yprobdf.to_csv(fname_prefix+'ypred_prob',sep='\t',index=False)
    ypreddf = pd.DataFrame(data=ypred,columns=['YPRED'])
    ypreddf.to_csv(fname_prefix+'ypred',sep='\t',index=False)
    return ypred

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-model',action='store_true',help='Fit model with training data, then predict on test data.')
    parser.add_argument('--features',help='File name for the training data. If there are multiple feature sets, use comma to delimit.')
    parser.add_argument('--target',help='File name for the training data.')
    parser.add_argument('--test-features',help='File name for the testing data. If there are multiple, use comma to delimit.')
    parser.add_argument('--output-prefix',help='Output files prefix, not including left out chrom and method')
    parser.add_argument('--leave-out',help='The chromosome which is left out')
    parser.add_argument('--method',help='The method using which we fit the model. E.g.: OLS, Lasso, GBT, RF')
    parser.add_argument('--num-annots',help='Number of annotations. If there are multiple annotation sets, use comma to delimit.')
    parser.add_argument('--recompute',action='store_true',help='Force recompute regardless of existing files.')
    args = parser.parse_args()

    if args.fit_model:
        fit_main(args.features,args.target,args.test_features,args.output_prefix,args.leave_out,args.method,args.num_annots,args.recompute)
