#import standard libraries
import pandas as pd
import numpy as np
from pandas.core.arrays import categorical
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import joblib
import argparse
import logging

#import user defined libraries
from constants import *

def prepare_data(data_type = 'train',
                 input_data_path = '',
                 output_data_path = '',
                 artifacts_path = '',
                 categorical_features = [],
                 numerical_features = [],
                 target_col = '',
                 logger = None
                 ):
    
    #load data
    df = pd.read_csv(input_data_path)
    print('data shape {}'.format(df.shape))
    #standardize column names
    df.rename(columns = {col:col.replace(' ','_').lower() for col in df.columns},inplace = True)
    print('data columns ',df.columns)

    if data_type == 'train':
        #parse date time columns
        for col in DATETIME_FEATURES:
            df[col] = pd.to_datetime(df[col])
        
        #create the target variables
        #df['lastworkingdate'].bfill(axis ='rows',inplace = True)
        df['lastworkingdate'] = df.groupby(['emp_id'])['lastworkingdate'].bfill()
        df[target_col] = 0
        df[target_col][((df['lastworkingdate'] - df['mmm-yy']).dt.days<=6*30) & ((df['lastworkingdate'] - df['mmm-yy']).dt.days>0)] = 1

        #get last 8 ratings and business value per employee
        for i in range(1,9,1):
            df[f'rating_minus_{i}'] = df.groupby(['emp_id'])['quarterly_rating'].shift(i)
            df[f'business_minus_{i}'] = df.groupby(['emp_id'])['total_business_value'].shift(i)
        
        #get designation min and max rating, bv
        df['designation_max_rating'] = df.groupby(['designation'])['quarterly_rating'].max()
        df['designation_min_rating'] = df.groupby('designation')['quarterly_rating'].min()
        df['designation_max_bv'] = df.groupby('designation')['quarterly_rating'].max()
        df['designation_min_bv'] = df.groupby('designation')['quarterly_rating'].min()

        '''
        group = df.groupby(['designation','gender'],as_index = False)['quarterly_rating'].max()
        group = group.rename(columns = {'quarterly_rating':'designation_max_rating'})
        df = df.merge(group,on = ['designation','gender'])
        print(df.columns)
        group = df.groupby(['designation','gender'],as_index = False)['quarterly_rating'].min()
        group = group.rename(columns = {'quarterly_rating':'designation_min_rating'})
        df = df.merge(group,on = ['designation','gender'])
        group = df.groupby(['designation','gender'],as_index = False)['total_business_value'].max()
        group = group.rename(columns = {'total_business_value':'designation_max_bv'})
        df = df.merge(group,on = ['designation','gender'])
        group = df.groupby(['designation','gender'],as_index = False)['total_business_value'].min().rename({'total_business_value':'designation_min_bv'})
        group = group.rename(columns = {'total_business_value':'designation_min_bv'})
        df = df.merge(group,on = ['designation','gender'])
        '''
        #create tenure column
        df['tenure'] = (df['mmm-yy'] - df['dateofjoining']).dt.days

        #get scaled values of numeric features
        standard_scaler = StandardScaler().fit(df[numerical_features])
        numerical_features_df = pd.DataFrame(standard_scaler.transform(df[numerical_features]),columns = numerical_features)
        print('numerical_features_df data shape {}'.format(numerical_features_df.shape))

        #get dummies for categorical features
        ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse = False, drop = 'first').fit(df[categorical_features])
        categorical_features_df = pd.DataFrame(ohe_encoder.transform(df[categorical_features]),columns = ohe_encoder.get_feature_names_out(categorical_features))
        print('categorical_features_df data shape {}'.format(categorical_features_df.shape))
        
        #save our data transformers
        joblib.dump(ohe_encoder,artifacts_path+'ohe_obj.pkl')
        joblib.dump(standard_scaler,artifacts_path+'standard_scaler.pkl')
    
    elif data_type == 'test':
        #no logic required here as we need to create the predictions from the train dataset
        print('not needed as there is no separate test dataset')
    
    #combine all individual datasets together
    out_df = pd.concat([numerical_features_df,categorical_features_df,df[[target_col]+['designation_max_rating','designation_min_rating','designation_max_bv','designation_min_bv']+PRIMARY_KEYS+[f'rating_minus_{i}' for i in range(1,9,1)] + [f'business_minus_{i}' for i in range(1,9,1)]]],axis = 1).fillna(0)

    print('out_df data shape {}'.format(out_df.shape))

    #write out the file to disk
    out_df.to_csv(output_data_path,index = False)
    print('writing to disk')
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Prep code')
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_data_path', type=str)
    parser.add_argument('--artifact_path', type=str)
    args = parser.parse_args()

    '''
    python3 prepare_data.py --data_type=train \
                            --input_data_path=/home/code/data/train_MpHjUjU.csv \
                            --output_data_path=/home/code/data/processed_train.csv\
                            --artifact_path=/home/code/artifacts/
    
    python3 prepare_data.py --data_type=test \
                            --input_data_path=/home/code/data/test_hXY9mYm.csv \    
                            --output_data_path=/home/code/data/processed_test.csv\
                            --artifact_path=/home/code/artifacts/
    '''

    logger = logging.getLogger(__name__)
    prepare_data(data_type = args.data_type,
                 input_data_path = args.input_data_path,
                 output_data_path = args.output_data_path,
                 artifacts_path = args.artifact_path,
                 categorical_features = CATEGORICAL_FEATURES,
                 numerical_features = NUMERICAL_FEATURES,
                 target_col='target',
                 logger = logger
                 )