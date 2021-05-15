# _tools.py
#
# Various tools for measpy
#
# (c) OD, 2021

import csv

def csv_to_dict(filename):
    """ Conversion from a CSV (produced by the class Measurement) to a dict
          Default separator is (,)
          First row is the key string
          The value is a list
    """
    dd={}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dd[row[0]]=row[1:]
    return dd

def convl(fun,xx):
    if type(xx)==list:
        yy=list(map(fun,xx))
    else:
        yy=fun(xx) 
    return yy

def convl1(fun,xx):
    if type(xx)==list:
        yy=fun(xx[0])
    else:
        yy=fun(xx) 
    return yy

def add_step(a,b):
    return a+'\n -->'+b
    
