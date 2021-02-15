from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
pd.set_option('display.max_columns', 100)
import os
import seaborn as sns
sns.set()
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
for dirname, _, filenames in os.walk('Predicting_Coupon_Redemption'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_df = pd.read_csv('Data/train/train.csv')
campaign_data_df = pd.read_csv('Data/train/campaign_data.csv')
coupon_item_mapping_df = pd.read_csv('Data/train/coupon_item_mapping.csv')
customer_demographics_df = pd.read_csv('Data/train/customer_demographics.csv')
customer_transaction_data_df = pd.read_csv('Data/train/customer_transaction_data.csv')
item_data_df = pd.read_csv('Data/train/item_data.csv')
test_df = pd.read_csv('Data/test.csv')
sub_df = pd.read_csv('Data/sample_submission.csv')

########################
# Feature Engineering in campaign_data.csv data.
campaign_data_df['start_date'] = pd.to_datetime(campaign_data_df['start_date'], format='%d/%m/%y', dayfirst=True)
campaign_data_df['end_date'] = pd.to_datetime(campaign_data_df['end_date'], format='%d/%m/%y', dayfirst=True)
campaign_data_df['diff_d'] = (campaign_data_df['end_date'] - campaign_data_df['start_date']) / np.timedelta64(1, 'D')
campaign_data_df.drop(['start_date','end_date'], axis=1, inplace=True)

# No Feature Engineering required for the train.csv data.


# No Feature Engineering required for the item_data.csv data.


# No Feature Engineering required for the coupon_item_mapping.csv data.


# Feature Engineering in customer_demographics.csv data.
lb = LabelEncoder()
customer_demographics_df['age_range'] = lb.fit_transform(customer_demographics_df['age_range'])

# customer_transaction_data.csv
customer_transaction_data_df['date'] = pd.to_datetime(customer_transaction_data_df['date'], format='%Y-%m-%d')
customer_transaction_data_df['date_d'] = customer_transaction_data_df['date'].dt.day.astype('category')
customer_transaction_data_df['date_m'] = customer_transaction_data_df['date'].dt.month.astype('category')
customer_transaction_data_df['date_w'] = customer_transaction_data_df['date'].dt.week.astype('category')
customer_transaction_data_df.drop(['date'], axis=1, inplace=True)

tgroup = customer_transaction_data_df.groupby(['customer_id']).sum().reset_index()

##################################

##
train_campaign_data = pd.merge(train_df,campaign_data_df,on='campaign_id',how="left")
test_campaign_data= pd.merge(test_df,campaign_data_df,on='campaign_id',how="left")

## Coupon Item Mapping.
coupon_item_mapping_item_data = pd.merge(coupon_item_mapping_df, item_data_df, on='item_id', how="left")
mci_group = pd.DataFrame()
mci_group[['coupon_id','no_of_items']] = coupon_item_mapping_item_data.groupby('coupon_id').count().reset_index()[
    ['coupon_id','item_id']]
mci_group[['brand_type','category']] = coupon_item_mapping_item_data.groupby('coupon_id').max().reset_index()[
    ['brand_type','category']]

##
mdtg = pd.merge(tgroup,customer_demographics_df,on='customer_id',how='outer')

#############################

## Merge all.
mergeddata = pd.merge(train_campaign_data, mdtg, on=['customer_id'], how='left')
mergeddata = pd.merge(mergeddata, mci_group, on=['coupon_id'], how='left')

mergeddata2 = pd.merge(test_campaign_data, mdtg, on=['customer_id'], how='left')
mergeddata2 = pd.merge(mergeddata2, mci_group, on=['coupon_id'], how='left')
id_df = mergeddata2['id']
mergeddata.drop(['id'],axis=1,inplace=True)
mergeddata2.drop(['id'],axis=1,inplace=True)

##############################

# Checking Missing values.
print(mergeddata.isnull().sum())
mergeddata.drop(['no_of_children','age_range','marital_status','rented','family_size','income_bracket'], axis=1, inplace=True)
print(mergeddata.isnull().sum())

print(mergeddata2.isnull().sum())
mergeddata2.drop(['no_of_children','age_range','marital_status','rented','family_size','income_bracket'], axis=1, inplace=True)
print(mergeddata2.isnull().sum())

#############################

cols = ['campaign_type','brand_type','category']

# Feature Encoding.
lb = LabelEncoder()
for i in cols:
    mergeddata[i] = lb.fit_transform(mergeddata[i])

for i in cols:
    mergeddata2[i] = lb.fit_transform(mergeddata2[i])

#############################

X = mergeddata.drop(['redemption_status'],axis=1)
Y = mergeddata['redemption_status']

#############################

## Handling class imbalance.

## Technique - 1.
fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
i = 1
auc = []

for train_index, test_index in fold.split(X, Y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    mergeddata2 = scaler.transform(mergeddata2)

    m = LGBMClassifier(random_state=80) # 93.23
    m.fit(x_train, y_train)
    pred_prob1 = m.predict_proba(x_val)
    auc.append(roc_auc_score(y_val, pred_prob1[:, 1]))
    i = i + 1

print("AUC Score")
print(sum(auc)/10)

## Technique-2.
# SMOTE.
from imblearn.over_sampling import SMOTE
from collections import Counter

# Oversample the dataset.
oversample = SMOTE()
X,Y = oversample.fit_resample(X,Y)
counter = Counter(Y)
print(counter)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 101)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_val = scaler.transform(X_test)

classifier_o = AdaBoostClassifier(random_state=20)  # 98.11
classifier_o.fit(X_train,y_train)
pred_prob1 = classifier_o.predict_proba(X_test)
print(roc_auc_score(y_test, pred_prob1[:, 1]))

classifier_o = GradientBoostingClassifier(random_state=30)  # 98.74
classifier_o.fit(X_train,y_train)
pred_prob1 = classifier_o.predict_proba(X_test)
print(roc_auc_score(y_test, pred_prob1[:, 1]))

classifier_o = LGBMClassifier(random_state=10)  # 99.78
classifier_o.fit(X_train,y_train)
pred_prob1 = classifier_o.predict_proba(X_test)
print(roc_auc_score(y_test,pred_prob1[:,1]))

######################

# LGBM Hyperparameter Tuning using Grid Search.
from sklearn.model_selection import GridSearchCV
param_grid = {'learning_rate': [0.1,0.5],
                 'max_depth': [4,5,6],
                 'num_leaves': [10,20],
                 'feature_fraction': [0.6,0.8],
                 'subsample': [0.2,0.6],
                 'objective': ['binary'],
              'metric': ['auc'],
              'is_unbalance':[False],
              'boosting':['gbdt'],
              'num_boost_round':[100],
              'early_stopping_rounds':[30]}

# Build and fit the GridSearchCV
grid = GridSearchCV(estimator=classifier_o, param_grid=param_grid,
                    cv=10,verbose=10)

grid_results = grid.fit(X_train,y_train,eval_set = (X_test,y_test))

# Summarize the results in a readable format.
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

#######################

classifier_o = LGBMClassifier(random_state=10,boosting='gbdt',feature_fraction=0.8,is_unbalance=False,learning_rate=0.5,max_depth=6,metric='auc',num_boost_round=100,num_leaves=20,objective='binary',subsample=0.2)  # 99.78
classifier_o.fit(X_train,y_train)
pred_prob1 = classifier_o.predict_proba(X_test)
print(roc_auc_score(y_test,pred_prob1[:,1]))  # 99.89

pred = classifier_o.predict(mergeddata2)
pred = pd.Series(pred)
data = {'id': id_df,
        'redemption_status':pred}
df = pd.DataFrame(data)
df.to_csv("final_predictions.csv")

###############################
