import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import pickle

# modelling imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC

#from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import  confusion_matrix
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, 
                              AdaBoostClassifier, BaggingRegressor)

import statsmodels.formula.api as sm

import xgboost as xgb
import re
import string 

# =============================================================================
# Creating PostgreSQL database on AWS EC2 instance
# =============================================================================

CREATE DATABASE yelp2
\connect yelp2


CREATE TABLE  yelp_business (
	    business_id VARCHAR(50) NOT NULL,
	    name       TEXT NOT NULL,   
	    neighborhood TEXT DEFAULT NULL,
	    address    TEXT DEFAULT NULL,
        city        TEXT DEFAULT NULL,
	    state      TEXT DEFAULT NULL,         
        postal_code TEXT DEFAULT NULL,
        latitude    FLOAT DEFAULT NULL,
        longitude   FLOAT DEFAULT NULL,
        stars       FLOAT DEFAULT NULL,
        review_count INT DEFAULT NULL,
        is_open     INT DEFAULT NULL,
        categories  TEXT DEFAULT NULL,
        
	    PRIMARY KEY (business_id)
    );

COPY yelp_business FROM '/home/ubuntu/yelp/yelp_business.csv' DELIMITER ',' CSV HEADER;


CREATE TABLE IF NOT EXISTS yelp_review (
        review_id VARCHAR(50) NOT NULL,
        user_id VARCHAR(50) NOT NULL,
        business_id VARCHAR(50) NOT NULL,
        stars INT DEFAULT NULL,
        date TEXT DEFAULT NULL,
        text TEXT DEFAULT NULL,
        useful INT DEFAULT NULL,
        funny INT DEFAULT NULL,
        cool INT DEFAULT NULL,

	    PRIMARY KEY (review_id)
    );

COPY yelp_review FROM '/home/ubuntu/yelp/yelp_review.csv' DELIMITER ',' CSV HEADER;


CREATE TABLE IF NOT EXISTS yelp_user (
        user_id VARCHAR(50) NOT NULL,
        name TEXT DEFAULT NULL,
        review_count INT DEFAULT NULL,
        yelping_since TEXT DEFAULT NULL,
        friends TEXT DEFAULT NULL,
        useful INT DEFAULT NULL,
        funny INT DEFAULT NULL,
        cool INT DEFAULT NULL,
        fans INT DEFAULT NULL,
        elite TEXT DEFAULT NULL,
        average_stars FLOAT DEFAULT NULL,
        compliment_hot INT DEFAULT NULL,
        compliment_more INT DEFAULT NULL,
        compliment_profile INT DEFAULT NULL,
        compliment_cute INT DEFAULT NULL,
        compliment_list INT DEFAULT NULL,
        compliment_note INT DEFAULT NULL,
        compliment_plain INT DEFAULT NULL,
        compliment_cool INT DEFAULT NULL,
        compliment_funny INT DEFAULT NULL,
        compliment_writer INT DEFAULT NULL,
        compliment_photos INT DEFAULT NULL,

	    PRIMARY KEY (user_id)
    );

COPY yelp_user FROM '/home/ubuntu/yelp/yelp_user.csv' DELIMITER ',' CSV HEADER;




# Postgres info to connect to local yelp database
connection_args = {
    'host': 'localhost',  # We are connecting to our _local_ version of psql
    'dbname': 'yelp',    # DB that we are connecting to
    'port': 5432          # port we opened on AWS
}
connection = pg.connect(**connection_args)



# Filter out non-restuarants 
# inner join yelp_user  with restaurant table and yelp_review table
q_join02 = """
            SELECT * 
            FROM ((SELECT * FROM yelp_business
            WHERE categories LIKE '%Restaurant%') AS yelp_restaurants
            INNER JOIN yelp_review ON yelp_restaurants.business_id = yelp_review.business_id) 
            AS yelp_rests_reviews
            INNER JOIN yelp_user ON yelp_rests_reviews.user_id = yelp_user.user_id;
           """
#Full dataframe
df_full = pd_sql.read_sql(q_join02, connection)


# Columns to drop: neighborhood
df_full.drop(['neighborhood'], axis=1, inplace=True)

# Rename columns to eliminate some duplicate names
df_full.columns = ['business_id','bus_name','address','city','state','postal_code',
                   'latitude','longitude','bus_stars_avg','bus_rev_count','is_open',
                   'categories','review_id','user_id','business_id','rev_stars',
                   'rev_date','rev_text','rev_useful','rev_funny','rev_cool',
                   'user_id','user_name','user_rev_count','yelping_since','friends',
                   'user_useful','user_funny','user_cool','user_fans','user_elite',
                   'user_avg_stars','compliment_hot','compliment_more','compliment_profile',
                   'compliment_cute','compliment_list','compliment_note','compliment_plain',
                   'compliment_cool','compliment_funny','compliment_writer','compliment_photos']


# Drop repeated business_id and user_id columns 
df_full.columns.values
df_full2 = df_full.loc[:,~df_full.columns.duplicated()]
df_full2.columns.values

# Drop any rows with nan (only dropped 1134 rows)
df_full2 = df_full2.dropna() 


# save df_full2 as csv 
df_full2.to_csv('df_full2.csv')
# open df_full2  ( in User/MJM... )
df_full2 = pd.read_csv('df_full2.csv')
df_full2.drop('Unnamed: 0',axis=1,inplace=True)


# =============================================================================
# # I Randomly selected 100000 rows to work with as a subset
# # After testing  with subset I worked with full set on an AWS EC2 instance
# From here on the code should work locally with a subset or with full data 
# =============================================================================

# 10,000 subset
df_10k = df_full2.sample(n=10000)

# 100,000 subset
df_10k_p = df_full2.sample(n=100000)

# for full dataset
df_10k = df_full2

# Pickle data (in project03 folder)
with open('df_10k.pkl', 'wb') as picklefile:
    pickle.dump(df_10k, picklefile) 
# open pickled df_10k
with open("df_10k.pkl", 'rb') as picklefile: 
    df_10k_p = pickle.load(picklefile) 

# NOTE: full dataframe may be too large to pickle. Use df_full2.csv
# in aws ubuntu terminal
df_10k_p = pd.read_csv('df_full2.csv')
df_10k_p.drop('Unnamed: 0',axis=1,inplace=True)


#c Check out data
describe=df_10k_p.describe()
df_10k_p.dtypes


# =============================================================================
#  Begin Feature Engineering and Feature Selection
# =============================================================================

# change rev_date to datetime object
df_10k_p['rev_date'] =  pd.to_datetime(df_10k_p['rev_date'])

# first feature, restaurant count
# How many restaurants/ is it a chain restaurant?
df_10k_p['bus_name'].value_counts()
df_10k_p = df_10k_p.join(df_10k_p.groupby('bus_name')['bus_name'].count(), on='bus_name', rsuffix='_c')

# distance from equator (absolute value of latitude)
df_10k_p['eq_dist'] = df_10k_p['latitude'].apply(pd.to_numeric, errors = 'coerce').abs()


# gb_bus -> group by business
# Average useful per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_useful'].mean(),
        on='business_id', rsuffix='_gb_bus')

# Average funny per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_funny'].mean(),
        on='business_id', rsuffix='_gb_bus')

# Average cool per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_cool'].mean(),
        on='business_id', rsuffix='_gb_bus')


# Average stars per business
#df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_stars'].mean(),
#        on='business_id', rsuffix='_gb_bus')
# rev_stars_gb_bus  = average rating per business (not rounded)



# max date in  review dates per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_date'].max(),
        on='business_id', rsuffix='_max')

# min date in  review dates per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_date'].min(),
        on='business_id', rsuffix='_min')

# Review date range column = max - min
df_10k_p['rev_date_range'] = df_10k_p['rev_date_max'] - df_10k_p['rev_date_min']
# not work in ubuntu


# text review length (characters)
df_10k_p['rev_len'] = df_10k_p['rev_text'].str.len()
# average review length (characters) per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_len'].mean(),
        on='business_id', rsuffix='_avg')

# concatenation of all review text per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['rev_text'].sum(),
        on='business_id', rsuffix='_cat')
# Seems to have worked??


# Pickle 10k subset (in project03 folder) version 2
with open('df_10k2.pkl', 'wb') as picklefile:
    pickle.dump(df_10k_p, picklefile) 
# open pickled df_10k
with open("df_10k2.pkl", 'rb') as picklefile: 
    df_10k_p = pickle.load(picklefile) 

# =============================================================================
# # More engineering using yelp_user information
# =============================================================================

# Average user review count per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_rev_count'].mean(),
        on='business_id', rsuffix='_gb')

df_10k_p = df_10k_p.reset_index()


# Average user useful/funny/cool sent by user
#df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_useful','user_funny','user_cool'].mean(),
#        on='business_id', rsuffix='_avg')

df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_useful'].mean(),
        on='business_id', rsuffix='_avg')

df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_funny'].mean(),
        on='business_id', rsuffix='_avg')

df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_cool'].mean(),
        on='business_id', rsuffix='_avg')

df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_fans'].mean(),
        on='business_id', rsuffix='_avg')


df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_avg_stars'].mean(),
        on='business_id', rsuffix='_avg')


# create new column, which is the differential between a user's overall average 
# and actual vote for a given business (actual - average)
df_10k_p['user_star_diff'] = df_10k_p['rev_stars'] - df_10k_p['user_avg_stars']
# now create average user_star_diff per business
df_10k_p = df_10k_p.join(df_10k_p.groupby('business_id')['user_star_diff'].mean(),
        on='business_id', rsuffix='_avg')


# Pickle 10k subset (in project03 folder) version 3
with open('df_10k3.pkl', 'wb') as picklefile:
    pickle.dump(df_10k_p, picklefile) 
# open pickled df_10k
with open("df_10k3.pkl", 'rb') as picklefile: 
    df_10k_p = pickle.load(picklefile) 

# =============================================================================
# Most features created. Reduce df to 1 busines per row
# Then cut some fat, remove obviously unneeded columns
# NOTE: new df name df_bus
# =============================================================================

df_bus = df_10k_p.drop_duplicates('business_id').copy()

# Drop irreleveant columns (remove compliment_photo)
df_bus.drop(['review_id','user_id','rev_stars','rev_date','rev_text','rev_useful',
             'rev_funny','rev_cool','user_name','user_rev_count', 'yelping_since',
             'friends','user_useful','user_funny','user_cool','user_fans','user_elite',
             'user_avg_stars','compliment_hot','compliment_more','compliment_profile',
             'compliment_cute','compliment_list','compliment_note','compliment_plain',
             'compliment_cool','compliment_funny','compliment_writer',
             'rev_len','user_star_diff'], axis=1, inplace=True)

# for AWS    
df_bus.drop(['review_id','user_id','rev_stars','rev_date','rev_text','rev_useful',
             'rev_funny','rev_cool','user_name','user_rev_count', 'yelping_since',
             'friends','user_useful','user_funny','user_cool','user_fans','user_elite',
             'user_avg_stars','compliment_hot','compliment_more','compliment_profile',
             'compliment_cute','compliment_list','compliment_note','compliment_plain',
             'compliment_cool','compliment_funny','compliment_writer',
             'rev_len','user_star_diff'], axis=1, inplace=True)
    
    
# add new column in which outcome is flipped
df_bus['is_open_r'] = df_bus['is_open'].replace({0:1, 1:0})


# =============================================================================
# =============================================================================
# # # SAVE AS CSV IN AWS at this point
# =============================================================================
# =============================================================================
df_bus.to_csv('df_bus_full.csv')

# open csv
df_bus = pd.read_csv('df_bus_full.csv')
df_bus.drop('Unnamed: 0',axis=1,inplace=True)


# Pickle 10k subset (in project03 folder) version 4
with open('df_bus_full.pkl', 'wb') as picklefile:
    pickle.dump(df_bus, picklefile) 
# open pickled df_10k
with open("df_bus_full.pkl", 'rb') as picklefile: 
    df_bus = pickle.load(picklefile) 
    
# =============================================================================
# =============================================================================
# =============================================================================
# # ***** 
# #script for adding in multiple predictors, including nlp on text
# # (start on AWS here)
# =============================================================================
# =============================================================================
# =============================================================================

df_bus = pd.read_csv('df_bus_full.csv')
df_bus.drop('Unnamed: 0',axis=1,inplace=True)

df_bus = df_bus.dropna()
df_bus = df_bus.reset_index(drop=True)


# =============================================================================
# Clean concatenated review texts in df_bus['rev_text_cat']
# =============================================================================

alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

df_bus['rev_text_cat'] = df_bus.rev_text_cat.map(alphanumeric).map(punc_lower)



# =============================================================================
# Create feature matix and outcome column
# =============================================================================

# Features selected ater testing with cross validation
X_other = df_bus.loc[ : , ['bus_rev_count','bus_name_c','eq_dist',
                           'rev_len_avg',
                           'user_rev_count_gb',
                           'user_star_diff_avg','bus_stars_avg']]          
# Scale non nlp features
scaler = StandardScaler().fit(X_other)
X_other_scaled = pd.DataFrame(scaler.transform(X_other))

# Text for NLP 
X_nlp = df_bus['rev_text_cat']

# concat scaled feature with text column
X_full = pd.concat([X_other_scaled, X_nlp], axis=1) 
# could also try numpy hstack /vstack


# Outcome variable, 0 = is open, 1 = closed
y = df_bus['is_open_r']


# =============================================================================
#  train test split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2) 

# =============================================================================
# td-idf vectorizer for NLP
# =============================================================================
# increasing max features improves performance, levels out around 8-10k
tfidfconverter = TfidfVectorizer( max_features=8000,max_df=0.50,
                                 stop_words='english', ngram_range=(1, 1))
                                      # ngram_range=(1, 2)
                                      # max_features=8000,
                                      # min_df=1,

                
# use fit_transform on training.  use transform on test
# TRAINING
X_nlp_only = pd.DataFrame(tfidfconverter.fit_transform(X_train['rev_text_cat']).toarray(),
                          columns=tfidfconverter.get_feature_names())

# concat normal features with NLP matrix
X_train = X_train.reset_index(drop=True)
X_full_train = pd.concat([X_train.loc[:, X_train.columns != 'rev_text_cat'], X_nlp_only], axis=1) 
X_full_train = X_full_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)



# TEST
X_nlp_only_test = pd.DataFrame(tfidfconverter.transform(X_test['rev_text_cat']).toarray(),
                          columns=tfidfconverter.get_feature_names())

X_test = X_test.reset_index(drop=True)
X_full_test = pd.concat([X_test.loc[:, X_test.columns != 'rev_text_cat'], X_nlp_only_test], axis=1) 


# =============================================================================
# # Balance/oversample after test/train/split. on training data
# =============================================================================

ros = RandomOverSampler(random_state=0, ratio=0.5) 
X_res, y_res = ros.fit_sample(X_full_train, y_train)

# =============================================================================
# Logistic Regression
# (tried many other models (e.g., naive bayes, decision tree, random forest)
#  logistic regression performed best so only showing that model below)
# =============================================================================

# min_df=0, max_df=0.50, (ros = 0.50)
lr04 = LogisticRegression(C=1).fit(X_res, y_res) 
lr04.score(X_res, y_res)        # 0.84   AWS .84
lr04.score(X_full_test, y_test) # 0.83   AWS .83

lr04_y_pred = lr04.predict(X_res)
lr04_y_pred_test = lr04.predict(X_full_test)

precision_score(y_res, lr04_y_pred)        # .84  AWS .84
precision_score(y_test, lr04_y_pred_test)  # .70  AWS .72

recall_score(y_res, lr04_y_pred)           # .65  AWS .65
recall_score(y_test, lr04_y_pred_test)     # .58  AWS .58

f1_score(y_res, lr04_y_pred)               # .73  AWS .72
f1_score(y_test, lr04_y_pred_test)         # .63  AWS .64

# Confusion matrix AWS
lr_cm = confusion_matrix(y_test, lr04.predict(X_full_test))
lr_cm = ([[7430,  635],[1196, 1645]])


sns.heatmap(lr_cm, cmap=plt.cm.Reds, annot=True, square=True, fmt='g',
           xticklabels=['in business','closed'],
           yticklabels=['in business','closed'])
plt.xlabel("Predicted", fontweight='bold',size=10)
plt.ylabel("Actual", fontweight='bold',size=10)
plt.title("Logistic Regression, Test Data\nAccuracy = .83, Recall = .58,\nPrecision = .72, F1 = .64", 
          fontweight='bold',size=12)



#LogisticRegression
scores = cross_validate(lr04, X_res, y_res, cv=5,
                        scoring=['accuracy','precision','recall','f1','roc_auc'])
scores_mean = np.mean(pd.DataFrame(scores))
scores_mean


# =============================================================================
# statsmodels
# =============================================================================



import statsmodels.formula.api as sm
 
model = sm.Logit(y_res, X_res)
result = model.fit()
result.summary()
result.summary().tables[1].as_html()


results_as_html = result.summary().tables[1].as_html()
p03sm = pd.read_html(results_as_html, header=0, index_col=0)[0]



model2 = sm.Logit(y_train, X_full_train)
result2 = model2.fit()
result2.summary()
result2.summary().tables[1].as_html()

results_as_html2 = result2.summary().tables[1].as_html()
p03sm2 = pd.read_html(results_as_html2, header=0, index_col=0)[0]
p03sm2.to_csv('p03statsmodelsdf.csv')



indices = np.argsort(tfidfconverter.idf_)[::]
features = tfidfconverter.get_feature_names()
top_n = 20
top_features = [features[i] for i in indices[:top_n]]
print (top_features)

# some top words in statsmodels logistic regression
"options", z = -5.8, p <.001
"welcoming", z = -5.4, p < .001
"beautiful", z = -4.7, p < .001
"sad", z = 7.0,



