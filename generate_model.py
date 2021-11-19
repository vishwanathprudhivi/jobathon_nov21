import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from extra_keras_metrics import get_minimal_multiclass_metrics
from sklearn.metrics import average_precision_score,accuracy_score,f1_score,precision_recall_curve
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#import user defined libraries
from constants import *

#set seed
np.random.seed(2021)

train_df  = pd.read_csv(PROCESSED_TRAIN_PATH)

feature_cols = ['salary','age','total_business_value','quarterly_rating','designation_2','designation_3','designation_4',
                'designation_5','gender_Male','education_level_College','education_level_Master'] \
                + [col for col in train_df.columns if 'joining_designation' in col]
                #+ [f'rating_minus_{i}' for i in range(1,5,1)] \        
                #+ [col for col in train_df.columns if 'city' in col] \
                #+ [f'business_minus_{i}' for i in range(1,5,1)]

#exclude data after 2017 q2 in training as the 0 labels are affected by the lack of actuals for 2018 q1 and q2
#we may want to include only the positive labels beyond this point in our training dataset
x = pd.concat([train_df[feature_cols][train_df['mmm-yy']<'2017-06-01'],train_df[feature_cols][(train_df['mmm-yy']>='2017-06-01') & (train_df[TARGET]==1)]],axis = 0)
y = pd.concat([train_df[TARGET][train_df['mmm-yy']<'2017-06-01'],train_df[TARGET][(train_df['mmm-yy']>='2017-06-01') & (train_df[TARGET]==1)]],axis = 0)
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size = 0.2)

def get_model_dl(input_shape,output_shape,x_train,y_train):
    l0 = layers.Input(shape = input_shape)
    l1 = layers.Dense(256,activation = 'relu')(l0)
    l5 = layers.Dense(256,activation = 'relu')(l1)
    l2 = layers.Dense(128,activation = 'relu')(l5)
    l3 = layers.Dense(64,activation = 'relu')(l2)
    l4 = layers.Dense(32,activation = 'relu')(l3)
    output = layers.Dense(output_shape,activation = 'sigmoid')(l4)
    model = Model([l0], [output])
    LEARNING_RATE = 0.001
    optimizer = Adam(lr = LEARNING_RATE)
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics = ['accuracy','AUC'])
    model.fit(x_train,y_train,batch_size = 50, nb_epoch= 15,
                        verbose=1, validation_data=(x_val, y_val),
                        shuffle = True)
    return model


#model training and evalution
dl_model = get_model_dl(len(feature_cols),1,x_train,y_train)
train_preds = dl_model.predict(x_train).round()
eval_preds = dl_model.predict(x_val).round()

print('train accuracy score = ',accuracy_score(y_train,train_preds),'test accuracy score = ',accuracy_score(y_val,eval_preds))
print('train f1 score = ',f1_score(y_train,train_preds),'test f1 score = ',f1_score(y_val,eval_preds))



#model prediction on test data. the test data will most likely contain employee ids that have not resigned
#till now, so we can safely pick those employee records for the latest period of yyy-mm from the training dataset for inference 
test_df = pd.read_csv(RAW_TEST_PATH)
test_x = train_df.sort_values(by=['emp_id','mmm-yy']).groupby(['emp_id'],as_index = False).last()
test_df = test_df.merge(test_x[feature_cols+['emp_id','mmm-yy']],how='inner',left_on='Emp_ID',right_on='emp_id')
test_preds = dl_model.predict(test_df[feature_cols])
test_df['Target'] = test_preds.round()

test_df[['Emp_ID','Target']].to_csv(PREDICTION_FILE_PATH,index = False)
#test_df.to_csv(PREDICTION_FILE_PATH,index = False)