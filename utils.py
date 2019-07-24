import pandas as pd
import numpy as np
import argparse
import sys
import warnings
from functools import partial
import pdb

def check_shape(df1,df2,axis):
    '''Checking if the shape of two dfs match
    along the specified axis
    '''
    if df1.empty or df2.empty:
        if df1.empty:
            warnings.warn('Dataframe in the first argument is empty')
        if df2.empty:
            warnings.warn('Dataframe in the second argument is empty')
        return True
    if not df1.empty and not df2.empty:
        shape1 = df1.shape
        shape2 = df2.shape
        if axis=='row':
            if shape1[0]==shape2[0]:
                return True
            else:
                return False
        elif axis=='column':
            if shape1[1]==shape2[1]:
                return True
            else:
                return False


def containList(l1,l2):
    '''Checking if list 1 contains in list 2
    '''
    return all(x in l2 for x in l1)


def nonInList(l1,l2):
    '''Checking if none of the elements in l1
    is in l2
    '''
    return all(x not in l2 for x in l2)


def dropCols(df, del_cols):
    '''Drop columns in del_cols which are also in df
    If not, then no error would be raised
    '''
    print('Dataframe has columns {}'.format(df.columns.tolist()))
    for col in (set(del_cols) & set(df.columns)):
        df = df.drop([col], axis=1)
        print('Dropping column {}'.format(col))
    return df

    
def strip_dir(fname):
    '''Strip directory information from file name
    '''
    return fname.split('/')[-1]


class memoize(object):
    '''cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    '''
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

