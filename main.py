import data
from text_classifier import classify_fracture, classify_avg, classify_fracture_avg_parallel
from concurrent.futures import ThreadPoolExecutor

TRAIN_DATA = 'kaggle/mimic_cxr_aug_train.csv'
VAL_DATA = 'kaggle/mimic_cxr_aug_validate.csv'

train_df, val_df = data.load_data(TRAIN_DATA, VAL_DATA)

data.concat_findings(train_df)
data.concat_findings(val_df)

for text in train_df['all_findings']:
    print(classify_fracture_avg_parallel(text))