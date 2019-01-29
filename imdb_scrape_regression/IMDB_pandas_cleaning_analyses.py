

import pandas as pd
import re
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


# IMPORT PICKLE FILE THEN ASSIGN AS DATAFRAME
with open("imdb_full_data.pkl", 'rb') as picklefile: 
    imdb_full_data_p = pickle.load(picklefile)  

df_imdb = pd.DataFrame(imdb_full_data_p)


# save df as csv 
df_imdb.to_csv('df_imdb.csv')
# open csv
df_imdb = pd.read_csv('df_imdb.csv')
df_imdb.drop('Unnamed: 0',axis=1,inplace=True)

# =============================================================================
# # BEGIN CLEANING
# =============================================================================

# check data types
df_imdb.dtypes
df_imdb.get_dtype_counts()

# remove extra whitespace at end of imdb_genre
df_imdb['imdb_genre'] = df_imdb['imdb_genre'].str.strip()
  

# change imdb total gross, imdb_num_votes, and imdb_ranking to numeric (float64)
df_imdb['imdb_gross'] = pd.to_numeric(df_imdb['imdb_gross'])
df_imdb['imdb_num_votes'] = pd.to_numeric(df_imdb['imdb_num_votes'])
# remove commas first
df_imdb['imdb_ranking'] = pd.to_numeric(df_imdb['imdb_ranking'].str.replace(',',''))


# SAVE POINT
df_imdb.to_csv('df_imdb02.csv')
df_imdb = pd.read_csv('df_imdb02.csv')
df_imdb.drop('Unnamed: 0',axis=1,inplace=True)


# =============================================================================
# # REGEX EXPRESSIONs
# =============================================================================
re_numbers = r'^[0-9]*$' # all numbers
re_nonnumbers = r'\D*'  # all nonnumbers 
re_fontfont = r'\<.*'  #everything after (and including) '<'
# =============================================================================
# 
# =============================================================================

# change imdb_year to numeric, 
df_imdb['imdb_year'] =  df_imdb['imdb_year'].str.replace(re_nonnumbers,'')
df_imdb['imdb_year'] = pd.to_numeric(df_imdb['imdb_year'])


## SAVE POINT (Proper datatypes)
#df_imdb.to_csv('df_imdb03.csv')
#df_imdb = pd.read_csv('df_imdb03.csv')
#df_imdb.drop('Unnamed: 0',axis=1,inplace=True)
###############################

# show counts of unique values in column
df_imdb['imdb_MPAA'].value_counts()
df_imdb['rating'].value_counts()
df_imdb['imdb_title'].value_counts()
df_imdb['imdb_genre'].value_counts()
df_imdb['distributor'].value_counts()
df_imdb['imdb_director'].value_counts()
df_imdb['imdb_gross'].value_counts()

# Drop Metascore because it has a lot of missing values
df_imdb = df_imdb.drop(['imdb_metascore'],axis=1)

# DROP ALL ROWS WITH NAN
df_imdb = df_imdb.dropna()

# NEXT, DATA WITH ONLY MAIN RATINGS, G, PG, PG-13, R, (NOT RATED)
# create mask
mpaa_mask5 = ((df_imdb['imdb_MPAA']=='G') | (df_imdb['imdb_MPAA']=='PG') | 
        (df_imdb['imdb_MPAA']=='PG-13') | (df_imdb['imdb_MPAA']=='R') | (df_imdb['imdb_MPAA']=='Not Rated'))

mpaa_mask4 = ((df_imdb['imdb_MPAA']=='G') | (df_imdb['imdb_MPAA']=='PG') | 
        (df_imdb['imdb_MPAA']=='PG-13') | (df_imdb['imdb_MPAA']=='R'))

## GO WITH 5 RATING VERSION
df_imdb = df_imdb[mpaa_mask5]

# LOG TRANSFORM imdb_gross
df_imdb["log_gross"] = df_imdb["imdb_gross"].apply(np.log)


## SAVE POINT (Proper datatypes)
#df_imdb.to_csv('df_imdb04.csv')
#df_imdb = pd.read_csv('df_imdb04.csv')
#df_imdb.drop('Unnamed: 0',axis=1,inplace=True)
###############################

# =============================================================================
#
#  Feature Engineering
#
# =============================================================================

# does movie contain ":"   0 = no, 1 = yes
df_imdb['colon']=99
for row in range(len(df_imdb)):
    if ":" in df_imdb.iloc[row,8]:
        df_imdb.iloc[row,11] = 1
    else:
        df_imdb.iloc[row,11] = 0

# does title start with "The "
df_imdb['the']=99
for row in range(len(df_imdb)):
    if "The " in df_imdb.iloc[row,8]:
        df_imdb.iloc[row,12] = 1
    else:
        df_imdb.iloc[row,12] = 0

        
# how many chacters in movie title?
df_imdb['title_length'] = -1
for row in range(len(df_imdb)):
    df_imdb.iloc[row,13] =  len(df_imdb.iloc[row,8])
        

# how many WORDS in movie title?
df_imdb['title_words'] = -1
for row in range(len(df_imdb)):
    df_imdb.iloc[row,14] =  len(df_imdb.iloc[row,8].split())


# IS MOVIE RATED R or NOT RATED? yes/no
df_imdb['r_or_nr']=99
for row in range(len(df_imdb)):
    if  df_imdb.iloc[row,0] == "R" or df_imdb.iloc[row,0] == "Not Rated":
        df_imdb.iloc[row,20] = 1
    else:
        df_imdb.iloc[row,20] = 0


## SAVE POINT (Proper datatypes)
#df_imdb.to_csv('df_imdb05.csv')
#df_imdb = pd.read_csv('df_imdb05.csv')
#df_imdb.drop('Unnamed: 0',axis=1,inplace=True)
##############################

# log of gross raw ( = imdb_gross*1000000).log
df_imdb['raw_gross'] = (df_imdb['imdb_gross']*1000000).apply(np.log)


#### Calculate average gross for each unique director
df_imdb = df_imdb.join(df_imdb.groupby('imdb_director')['imdb_gross',
                       'log_gross','raw_gross'].mean(), on='imdb_director', rsuffix='_r')
    

#### create count column. for number of movies director has
df_imdb = df_imdb.join(df_imdb.groupby('imdb_director')['imdb_director'].count(), on='imdb_director', rsuffix='_c')

#  NOTE TRY NOT TO USE CURRENT MOVIE AS PART OF director average gross 
# if director has one movie, impute median 
for row in range(len(df_imdb)):
    if df_imdb.iloc[row,19] == 1:
        df_imdb.iloc[row,18] = df_imdb['raw_gross'].median()      # raw_gross_r (log)
        df_imdb.iloc[row,16] =  df_imdb['imdb_gross'].median()     # imdb_gross_r (actual, in millions)


## SAVE POINT (Proper datatypes)
#df_imdb.to_csv('df_imdb06.csv')
#df_imdb = pd.read_csv('df_imdb06.csv')
#df_imdb.drop('Unnamed: 0',axis=1,inplace=True)
###############################
#
## SAVE POINT (Proper datatypes)
#df_imdb.to_csv('df_imdb07.csv')
#df_imdb = pd.read_csv('df_imdb07.csv')
#df_imdb.drop('Unnamed: 0',axis=1,inplace=True)


# =============================================================================
# USE GROUPBY TO GET MEANS/MEDIANSSTD/ETC OF KEY VARIABLES
# =============================================================================
gb_stats = pd.DataFrame(df_imdb.groupby('colon')['imdb_gross','imdb_rating','imdb_runtime',
                        'title_length'].median())
    
# Without : Mean gross = 29m(SD=51.3)(Med.=10.2), imdb rating=6.40(1.01)(Med.=6.5), runtime=107.8(19.3)(Med.=104), title_length=14.1(6.8)(Med.=13)
# With :    Mean gross = 66.5m(SD=105.4)(Med.=23.2), imdb rating=6.09(1.24)(Med.=6.2), runtime-108.8(21.7)(Med.=100), title_length=30.7(9.2)(Med.=13)             
    
gb_stats2 = pd.DataFrame(df_imdb.groupby('r_or_nr')['imdb_gross','imdb_rating','imdb_runtime',
                        'title_length'].median())
   
    
# SAVE POINT (Proper datatypes)
df_imdb.to_csv('df_imdb08.csv')
df_imdb = pd.read_csv('df_imdb08.csv')
df_imdb.drop('Unnamed: 0',axis=1,inplace=True)


# =============================================================================
# # =============================================================================
# # GRAPHS, EXPLORTORY DATA ANALYSIS
# # =============================================================================
# =============================================================================

sns.set_style("whitegrid")

# BOXPLOTS
sns.boxplot(data=df_imdb, x="colon", y="imdb_gross")
plt.ylim(0, 250)
plt.show()

sns.boxplot(data=df_imdb, x="colon", y="raw_gross")
plt.ylim(0, 250)
plt.show()

sns.boxplot(data=df_imdb, x="colon", y="log_gross", hue='imdb_MPAA')
plt.show()



### COLON IN TITLE, WITH VS WITHOUT
## points over boxplot (logged gross)
ax2 = sns.boxplot(data=df_imdb, x="colon", y="raw_gross")
sns.swarmplot(data=df_imdb, x="colon", y="raw_gross", size=1.3, color='black', edgecolor='white',alpha=1.0)
plt.xticks(df_imdb['colon'],["With :","Without :"],fontweight="bold",size=14)
plt.xlabel("")
plt.ylabel("Total Domestic Gross (log)", fontweight="bold",size=14)
plt.title("Total Domestic Gross\nTitles Without vs. With : ", size=18, fontweight="bold")

mybox11 = ax2.artists[0]
# Change the appearance of that box
mybox11.set_facecolor('#A2F5F3')
mybox11.set_edgecolor('gray')
mybox11.set_linewidth(1)

mybox22 = ax2.artists[1]
# Change the appearance of that box
mybox22.set_facecolor('#F9ACEF')
mybox22.set_edgecolor('gray')
mybox22.set_linewidth(1)


### RATING: G/PG/PG13 VS R/NOTRATED
ax3 = sns.boxplot(data=df_imdb, x="r_or_nr", y="raw_gross")
sns.swarmplot(data=df_imdb, x="r_or_nr", y="raw_gross", size=1.3, color='black', edgecolor='white',alpha=1.0)
plt.xticks(df_imdb['r_or_nr'],[".                                                                     G, PG, PG-13                                                R, Not Rated"],
           fontweight="bold",size=14)
plt.xlabel("")
plt.ylabel("Total Domestic Gross (log)", fontweight="bold",size=14)
plt.title("Total Domestic Gross by\n MPAA Rating", size=18, fontweight="bold")

mybox111 = ax3.artists[0]
# Change the appearance of that box
mybox111.set_facecolor('#ACBCF9')
mybox111.set_edgecolor('gray')
mybox111.set_linewidth(1)

mybox222 = ax3.artists[1]
# Change the appearance of that box
mybox222.set_facecolor('#FBA4C5')
mybox222.set_edgecolor('gray')
mybox222.set_linewidth(1)



# BOX PLOT OF RATING by TOTAL GROSS (logged)
ax = sns.boxplot(data=df_imdb, x="imdb_MPAA", y="raw_gross",showfliers=False)
sns.swarmplot(data=df_imdb, x="imdb_MPAA", y="raw_gross", size=1.1, 
              color='black', edgecolor='gray',alpha=1.0)
plt.xticks(size=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("Total Domestic Gross (log)", fontweight="bold",size=14)
plt.title("Total Domestic Gross by MPAA Rating", size=18, fontweight="bold")

mybox1 = ax.artists[0]
# Change the appearance of that box
mybox1.set_facecolor('orange')
mybox1.set_edgecolor('gray')
mybox1.set_linewidth(1)

mybox2 = ax.artists[1]
# Change the appearance of that box
mybox2.set_facecolor('#3FD4F7')
mybox2.set_edgecolor('gray')
mybox2.set_linewidth(1)

mybox3 = ax.artists[2]
# Change the appearance of that box
mybox3.set_facecolor('#3FF782')
mybox3.set_edgecolor('gray')
mybox3.set_linewidth(1)

mybox4 = ax.artists[3]
# Change the appearance of that box
mybox4.set_facecolor('red')
mybox4.set_edgecolor('gray')
mybox4.set_linewidth(1)

mybox5 = ax.artists[4]
# Change the appearance of that box
mybox5.set_facecolor('#B18BF8')
mybox5.set_edgecolor('gray')
mybox5.set_linewidth(1)

# Check out correlation matrix
df_corr=df_imdb.corr()

### SCATTERPLOTS ###
sns.scatterplot(data=df_imdb, x="imdb_runtime", y="raw_gross",
                edgecolor='none',s=7,alpha=0.4,color="brown")

sns.scatterplot(data=df_imdb, x="raw_gross_r", y="raw_gross",
                edgecolor='none',s=5,alpha=0.5)

# Director Film count in top 10,000 vs gross (logged)
sns.stripplot(data=df_imdb, x="imdb_director_c", y="raw_gross", jitter=1.1,color='#ED234B',
                edgecolor='brown',s=3,alpha=0.3)
plt.xticks(fontweight='bold')
plt.xlabel("Number of films in top 10,000",size=16,fontweight='bold')
plt.ylabel("Domestic Gross (log)",size=16,fontweight='bold')
plt.title("Director Films in Top 10,000",size=22, fontweight='bold')


# PAIR PLOTS
sns.pairplot(df_imdb, vars=['imdb_gross','imdb_gross_r',
                            'imdb_runtime','title_length'])

sns.pairplot(df_imdb, vars=['raw_gross','raw_gross_r',
                            'imdb_runtime','title_length'])
    
sns.pairplot(df_imdb, vars=['raw_gross','raw_gross_r',
                            'imdb_runtime','title_length','imdb_director_c'])


# =============================================================================
# ##### STATSMODEL regressions ######
# =============================================================================
# Use statsmodel first to explore stats for each predictor.
# After trying many variations of predictors (removed to keep the code clean)
# The model below was the strongest and most parsimonious

lm8 = smf.ols('raw_gross ~ imdb_runtime + r_or_nr +  + colon + raw_gross_r + imdb_director_c', data=df_imdb)
fit8 = lm8.fit()
fit8.summary()
# R2 = .483 
    
     
# =============================================================================
#     SK LEARN regressions  with PoltnomialFeatures and Lasso
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_imdb[['imdb_runtime','colon',
                                                        'raw_gross_r', 'r_or_nr','imdb_director_c']], 
                                                    df_imdb['raw_gross'], test_size=0.30)

m1 = LinearRegression()
m1.fit(X_train,y_train)
m1.score(X_train,y_train)
# R2 = .48
m1.score(X_test,y_test)
# R2 =.49


# TRY POLYNOMIAL FEATURES WITH imdb_runtime + colon + raw_gross_r + r_or_nr + imdb_director_c
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

p = PolynomialFeatures(degree=2)
m1.fit(p.fit_transform(X_train),y_train)
m1.score(p.transform(X_train),y_train)
# R2 = 0.50
m1.score(p.transform(X_test),y_test)
# R2 = 0.51

p.get_feature_names()
m1.coef_


# TRY LASSO 
m1 = LassoCV()
p = PolynomialFeatures(degree=2)
m1.fit(p.fit_transform(X_train),y_train)
m1.score(p.transform(X_train),y_train)
# LASSO R2 = 0.50
m1.score(p.transform(X_test),y_test)
# LASSO R2 = 0.49

p.get_feature_names()
m1.coef_

####  POLYNOMIAL FEATURES and LASSO DID NOT AFFECT MODEL SUBSTANTIALLY..
# SHOWED SOME POTENTIAL INTERACTIONS THOUGH??
# WORTH INVESTIGATED FURTHER IN FUTURE




