#import standard libraries
import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns

#import user defined libraries
from constants import PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH,TRAIN_REPORT_PATH,TEST_REPORT_PATH

#load data
train_df = pd.read_csv(PROCESSED_TRAIN_PATH)

#create a profile report for the data sets
train_report = pandas_profiling.ProfileReport(train_df, title="Train Data Report")
train_report.to_file(TRAIN_REPORT_PATH)
