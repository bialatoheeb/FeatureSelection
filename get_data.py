#!/bin/python3
import pandas as pd

def breast_cancer_data():
    columns = ['Clump Thickness', 'Uniformity of Cell Size',      
            'Uniformity of Cell Shape' , 'Marginal Adhesion',
           'Single Epithelial Cell Size',   'Bare Nuclei' ,  
            'Bland Chromatin', 'Normal Nucleoli',  'Mitoses',
            'class_label']
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df = pd.read_csv(url, names=columns)
    # Remove this feature since it has null values and this notebook is for showing feature_selection methods
    df.drop('Bare Nuclei', axis=1, inplace=True)  
    y = df['class_label']
    df.drop('class_label', axis=1, inplace=True)
    return df, y

def glass_data():
    glass_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
    df = pd.read_csv(glass_url, header=None,index_col=None)
    return df[df.columns[:-1]], df[df.columns[-1]]

