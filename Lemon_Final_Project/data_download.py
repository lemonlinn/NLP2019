# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:07:29 2019

@author: Lemon Lin Reimer
"""

#%% Scraping Data

from FinalCrawler import mycrawler
import pandas as pd

myfunc = mycrawler()
query = ["POV","Hidden Cam","Cuckold","VR"]

df = [myfunc.comments(q) for q in query]
rawdata = pd.concat(df, ignore_index = True)
data = myfunc.cleaner(rawdata)

myfunc.teardown()

#%% writing data to .csv

data.to_csv("newporn.csv", index = False)