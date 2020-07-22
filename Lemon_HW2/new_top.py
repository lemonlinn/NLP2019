# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 07:35:36 2019

@author: lemon
"""
import pandas as pd
from new_crawler import crawler

topics = ["vulfpeck","tank and the bangas","magic sword","louie zong"]

my_func = crawler()

df = [(my_func.write_crawl_results(the_query, 10)) for the_query in topics]
    
mydata = pd.concat(df, ignore_index = True)

print(mydata)