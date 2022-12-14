# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:55:14 2019

@author: austinpeel
"""
import pandas as pd
import pyodbc
from config import server, database,username, password

#function needs to autolookup for each each, Ideally pass a year and get df for that year, and a survey
def get_css():
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password+'')
    df = pd.read_sql("select * from CSS_2019_RAW_TMP where variable ='Q11.10' or variable ='Q12.19' or variable ='Q13.31' or variable ='Q14.7' ",cnxn)
    cnxn.close()
    return df