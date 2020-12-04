# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:31:55 2020

@author: user
"""

import pandas as pd
import numpy as np

diabetes = pd.read_csv('data/diabetes_csv.csv', header=None)
liver = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv', header=None)
ionosphere = pd.read_table('data/ionosphere_data.txt', sep=',', header=None).iloc[0,:]
ionos = []
for i in ionosphere:
    ionos.append(i.split(','))
ionosphere = pd.DataFrame(ionos)    
del ionos
