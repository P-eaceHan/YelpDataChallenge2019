'''
This file includes data cleaning and preprocessing
'''
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

def read_data(file_path):
	return pd.read_csv(file_path)

def split_data(data):
	return train_test_split(data)

data = read_data("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/review_sub.csv")
train_data, test_data = split_data(data)
del data
train_data.to_csv("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/train.csv", sep=",")
test_data.to_csv("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/test.csv", sep=",")