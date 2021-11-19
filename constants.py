mode = 'docker container'
docker_dir = '/home/code'
local_dir = '/Users/vishwanathprudhivi/Desktop/Code/jobathon_nov21'


if mode == 'docker container':
    RAW_TRAIN_PATH = f'{docker_dir}/data/train_MpHjUjU.csv'
    RAW_TEST_PATH = f'{docker_dir}/data/test_hXY9mYm.csv'
    TRAIN_REPORT_PATH = f'{docker_dir}/reports/train_report.html'
    TEST_REPORT_PATH = f'{docker_dir}/reports/test_report.html'
    PROCESSED_TRAIN_PATH = f'{docker_dir}/data/processed_train.csv'
    PROCESSED_TEST_PATH = f'{docker_dir}/data/processed_test.csv'
    ARTIFACTS_PATH = f'{docker_dir}/artifacts/'
    PREDICTION_FILE_PATH = f'{docker_dir}/data/prediction_file.csv'

elif mode == 'local':
    RAW_TRAIN_PATH = f'{local_dir}/data/train_MpHjUjU.csv'
    RAW_TEST_PATH = f'{local_dir}/data/test_hXY9mYm.csv'
    TRAIN_REPORT_PATH = f'{local_dir}/reports/train_report.html'
    TEST_REPORT_PATH = f'{local_dir}/reports/test_report.html'
    PROCESSED_TRAIN_PATH = f'{local_dir}/data/processed_train.csv'
    PROCESSED_TEST_PATH = f'{local_dir}/data/processed_test.csv'
    ARTIFACTS_PATH = f'{local_dir}/artifacts/'
    PREDICTION_FILE_PATH = f'{local_dir}/data/prediction_file.csv'

PRIMARY_KEYS = ['emp_id','mmm-yy','lastworkingdate','dateofjoining']

CATEGORICAL_FEATURES = ['designation','gender','city','education_level']
NUMERICAL_FEATURES = ['salary','age','total_business_value','quarterly_rating']
DATETIME_FEATURES = ['lastworkingdate','dateofjoining','mmm-yy']

TARGET = 'target'