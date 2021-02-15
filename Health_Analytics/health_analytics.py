from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
pd.set_option('display.max_columns', 100)
import os
import seaborn as sns
sns.set()
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
for dirname, _, filenames in os.walk('HealthAnalytics2'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder, StandardScaler

# %%

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')
sub_df = pd.read_csv('Data/sample_submission_lfbv3c3.csv')

# %%

# Training data.
print('Training data shape: ', train_df.shape)
train_df.head(5)

# Test data.
print('Test data shape: ', test_df.shape)
test_df.head(5)

# %%

# Null values and Data types.
print('Train Set')
print(train_df.info())
print('-------------')
print('Test Set')
print(test_df.info())

# %%

print(train_df.isnull().sum())
print(test_df.isnull().sum())

train_df['Bed Grade'].fillna(train_df['Bed Grade'].mode()[0], inplace=True)
train_df['City_Code_Patient'].fillna(train_df['City_Code_Patient'].mode()[0], inplace=True)

test_df['City_Code_Patient'].fillna(test_df['City_Code_Patient'].mode()[0], inplace=True)
test_df['Bed Grade'].fillna(test_df['Bed Grade'].mode()[0], inplace=True)

print(train_df.isnull().sum())
print(test_df.isnull().sum())

# %%

# Total number of Patients in the dataset(train+test)
print("Total Patients in Train set: ", train_df['patientid'].nunique())
print("Total Patients in Test set: ", test_df['patientid'].nunique())

# %%

print(train_df.columns)
print(test_df.columns)

# %%

for col in train_df.columns:
    print('Number of unique values of ' + col + ' column in train_df dataset are {} '.format(train_df[col].nunique()))
    print('The unique values of ' + col + ' column in train_df dataset are {} '.format(train_df[col].unique()))

for col in test_df.columns:
    print('Number of unique values of ' + col + ' column in train_df dataset are {} '.format(test_df[col].nunique()))
    print('The unique values of ' + col + ' column in train_df dataset are {} '.format(test_df[col].unique()))

print(train_df['Type of Admission'].value_counts(normalize=True))
print(train_df['Severity of Illness'].value_counts(normalize=True))
print(train_df['Department'].value_counts(normalize=True))

# %%

print(train_df.dtypes)
print(test_df.dtypes)

# %%.

# Target class distribution.
print(train_df['Stay'].value_counts())
print(train_df['Stay'].value_counts(normalize=True))
print(train_df['Stay'].unique())
print(train_df['Stay'].nunique())

le = LabelEncoder()
train_df['Stay'] = le.fit_transform(train_df['Stay'])
print("Label encoding")
print(train_df['Stay'])

# %%

cols = ['Hospital_type_code',
        'Hospital_region_code',
        'Department','Ward_Type','Ward_Facility_Code',
        'Type of Admission',
        'Severity of Illness','Age'
        ]

for col in cols:
    if train_df[col].dtype == object:
        print(col)
        lbl = LabelEncoder()
        train_df[col] = lbl.fit_transform(train_df[col])

for col in cols:
    if test_df[col].dtype == object:
        print(col)
        lbl = LabelEncoder()
        test_df[col] = lbl.fit_transform(test_df[col])

# Total visits of a patient to hospital, total no. of visitors to patient.
Encoding = train_df.groupby('patientid')['case_id'].count()
print(Encoding)

# %%
X = train_df.drop(['Stay', 'case_id', 'patientid'], axis=1)
Y = train_df['Stay']
test_X = test_df.drop(['case_id', 'patientid'], axis=1)

fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
i = 1
acc = []

for train_index, test_index in fold.split(X, Y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    test_X = scaler.transform(test_X)

    m = LGBMClassifier()
    m.fit(x_train, y_train)
    pred_y = m.predict(x_val)
    acc.append(accuracy_score(y_val, pred_y))
    i = i + 1


print(acc)
print(sum(acc) / 10)
