#!/usr/bin/env python
# coding: utf-8

# # Roberto Bertolini
# # Ames Housing Dataset

# ## Preliminary Steps: Reading Libraries & Training Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm 
import statsmodels.stats.api as sms
from matplotlib import pyplot
from scipy.stats import chi2_contingency
import itertools
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import permutation_test_score
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance


# In[2]:


train = pd.read_csv('train.csv') # Import the training data
print(train.head(15))
print(train.describe())


# ## Part 1 - Pairwise Associations Between Predictors

# ### Continuous Predictors

# In[3]:


# After consulting the data and the text description file provided, I started off with a seletion of features:

train_subset = ['LotArea','LotShape','HouseStyle','TotalBsmtSF','GrLivArea','OverallQual',
                'Neighborhood','Foundation','FullBath','SaleCondition','SalePrice', 'YrSold','MoSold']
train_10 = train[train_subset]
train_10['YrSold'] = train_10['YrSold'].astype(str) # A year is nominal so convert it to a string variable
train_10['MoSold'] = train_10['MoSold'].astype(str) # Month is nominal so convert it to a string variable
corr = train_10.corr() # Get the pearson correlation coefficient
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True).set_title('Heat Map of Pearson Correlation Coefficient')


# Pearson's correlation coefficient is only calculated for continuous predictors. For multi-class categories, there are more appropriate metrics to quantify the association between predictors such as Cramer's V (see below).
# 
# There is a strong linear association between the variables OverallQual and SalePrice (r = 0.79). As the overall quality of the house increases, the sale price of the house also increases. There is a weak, positive correlation between the number of full bathrooms in the house and the LotArea of the house (r = 0.11). Moreover, there were no negative correlations between the continuous predictors chosen. 

# ### Categorical Predictors

# Pearson's correlation coefficient is not appropriate for categorical predicors. Therefore, Cramer's V statistic was calculated - this is commonly used in statistics to measure the association between two categorical predictors

# In[4]:


# I needed to manually implement Cramer's V association matrix. 

# I first began by computing all pairwise combinations of the continuous predictors. 

cross_tabs = list(itertools.product(['YrSold','MoSold','LotShape','HouseStyle','Neighborhood','Foundation','SaleCondition'], 
                            ['YrSold','MoSold','LotShape','HouseStyle','Neighborhood','Foundation','SaleCondition']))

# Cramer's V statistic corrected for bias (see references at the end of this question section)

def cramers_corrected_stat(confusion_matrix):
    chi2_info = chi2_contingency(confusion_matrix)[0] # Compute the continguency table detailing the frequency of predictors
    n = confusion_matrix.sum().sum() # Return the overall mean
    phi2 = chi2_info/n
    i,j = confusion_matrix.shape # Extract the shape of the matrix
    phi2corr = max(0, phi2 - ((j-1)*(i-1))/(n-1)) 
    j_corr = j - ((j-1)**2)/(n-1)
    i_corr = i - ((i-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (j_corr-1), (i_corr-1)))

# Apply the formula from the function above to compute cramer's v between each pair of variables. 
cram = [] # Create an empty list to store cramer's v statistics
for i in range(0,49):
        confusion_matrix = pd.crosstab(train_10[cross_tabs[i][0]], train_10[cross_tabs[i][1]]) # Cross-tabulation of data values
        cramer_v = cramers_corrected_stat(confusion_matrix) # Apply the function above
        cram.append(cramer_v) # Append to the list

# Split the list into the entries which will form the columns of the matrix
col_1_cram = cram[0:7] # Column 1
col_2_cram = cram[7:14]# Column 2
col_3_cram = cram[14:21] # Column 3
col_4_cram = cram[21:28] # Column 4
col_5_cram = cram[28:35] # Column 5
col_6_cram = cram[35:42] # Column 6
col_7_cram = cram[42:49] # Column 7


# In[5]:


# Create a dataframe with the cramer's v columns
cramer_df = pd.DataFrame(col_1_cram, index =['YrSold', 'MoSold', 'LotShape', 'HouseStyle', 'Neighborhood', 'Foundaton', 'SaleCondition'], 
                                              columns =['YrSold'])
# Rename the columns
cramer_df['MoSold'] = col_2_cram
cramer_df['LotShape'] = col_3_cram
cramer_df['HouseStyle'] = col_4_cram
cramer_df['Neighborhood'] = col_5_cram
cramer_df['Foundation'] = col_6_cram
cramer_df['SaleCondition'] = col_7_cram


# In[6]:


# Plot Cramer's V Statistic Heat Map
sns.heatmap(round(cramer_df,2), xticklabels=cramer_df.columns,yticklabels=cramer_df.columns,annot=True).set_title('Heat Map of Cramers V Statistic')


# Cramer's V heat map reveals that categorical variables such as LotShape and MoSold are not associated with each other (Cramer's V: 0) while other predictors such as Neighborhood and Foundation are moderately associated with one another (Cramer's V: 0.42)

# References for Cramer's V:
# https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix/38032115

# ## Part 2 - A Selection of Informative Plots

# ### Plot 1

# In[7]:


# Boxplot of Overall House Quality and Log(Sale Price)
sns.set(style="whitegrid") # Set boxplot style
#ax = sns.boxplot(x=train_10["OverallQual"], y=np.log(train_10["SalePrice"]),order=["1","2","3","4","5","6","7","8","9","10"])
ax = sns.boxplot(x=train_10["OverallQual"], y=np.log(train_10["SalePrice"]))
plt.title('Boxplot of Overall Quality and Log(Sale Price)',size=15) # tile
plt.xlabel('Overall Quality',size=15) # x-label
plt.ylabel('Log(Sale Price)',size=15) # y-label
plt.xticks(size=15) # size of x-ticks
plt.yticks(size=15) # size of y-ticks


# The distribution of sale price is skewed to the right so a logarithm transformation was applied to this variable.
# 
# The boxplot shows that as the overall house quality increases, the logarithm of the sales price also increases. The range for each distribution is roughly equal across quality ranking 3-8. Moreover, there are several outliers below each boxplot. Therefore, even though some houses receive a good quality ranking (e.g. 5, 6, 7), some houses sell for a lower price than others with the same ranking. This means that there are other factors besides the quality ranking that impact the price of a house in Ames, Iowa.

# ### Plot 2

# In[8]:


# Scatterplot of Greater Living Area and Log(Sale Price) with the line of best fit drawn
plt.scatter(train_10['GrLivArea'], np.log(train_10['SalePrice']), s = .75,alpha=1) # Scatterplot
# Plot the line of best fit
plt.plot(np.unique(train_10['GrLivArea']), np.poly1d(np.polyfit(train_10['GrLivArea'], np.log(train_10['SalePrice']), 1))(np.unique(train_10['GrLivArea'])),color="black")
plt.text(100,14.10,'log(Sale Price) = 0.0005 + 0.11*GrLivArea',size=12) # Add the equation text
plt.title('Scatter Plot of Greater Living Area and Log(Sale Price)',size=15) # Title
plt.xticks(size=15) # X-ticks
plt.yticks(size=15) # Y-ticks
plt.xlabel('GrLivArea',size=15) # X-label
plt.ylabel('Log(Sale Price)',size=15) # Y-label
plt.show()


# Plot 2 shows that there is a strong linear relationship between the greater living area and log(sale price). The best fit line is shown in black. There are a few outlier points corresponding to a log(sale price) of 12.0 and a greater living area between 4000 and 5000. This plot reinforces the strong correlation between these variables from Question 1 (r = 0.71). 
# 
# Reference to construct plot:
# https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python

# ### Plot 3

# In[9]:


# Create a generic function to plot a bar chart and display the frequency on top of the bar.

def createBarChart(variable,x_axis_labels,title,y_limits,dataset):
    plt.figure(figsize=(10,6)) # Figure Size
    x_position = np.arange(len(x_axis_labels)) # Create the xticks
    plt.bar(x_position,dataset[variable].value_counts(dropna=False), align='center', edgecolor='black') # Include missing values 
    plt.xticks(x_position, x_axis_labels,rotation=90,size=15) # Rotate the x-ticks
    plt.yticks(size=15)
    plt.xlabel('Neighborhood',size=15) # Y-label
    plt.ylabel('Frequency',size=15) # Y-label
    plt.title(title,size=15) # Title
    plt.ylim(y_limits) # Y-limits

    for pos in ['right','top','bottom']:
        plt.gca().spines[pos].set_visible(False)
    
# Add the frequency count on the top of each barplot
    for i in range(len(x_axis_labels)):
        plt.text(x = x_position[i], 
                 y = dataset[variable].value_counts(dropna=False)[i]+2, 
                 s = dataset[variable].value_counts(dropna=False)[i],
                 size = 14, ha = 'center')
    plt.grid(axis = 'x', color ='white', linestyle='-',linewidth=8)
    plt.show()
    
createBarChart('Neighborhood',('North Ames', 'College Creek', 'Old Town', 'Edwards', 'Somerset', 'Gilbert',
       'Northridge Heights', 'Sawyer', 'Northwest Ames', 'Sawyer West', 'BrookSide', 'Crawford',
       'Mitchell', 'Northridge', 'Timberland', 'DOT & RR', 'Clear Creek', 'Stone Brook', 'S & W of ISU',
       'Bloomington Heights', 'Meadow Village', 'Briardale', 'Veenker', 'Northpark Villa', 'Bluestem'),
               'Frequency of Houses Sold in Each Neighborhood',[0,250],train_10)


# The third plot displays the freqency of the houses sold in each neighborhood in the training data set. North Ames had the largest amount of homes sold: 225 (15.4%) while in Bluestem, only 2 houses (0.1%) were sold. 

# ### Plot 4

# In[10]:


# Line graph of number of homes sold per month

train_plot4 = train[['MoSold','YrSold']] # Extract and sort the month and year sold columns.
train_plot4 = train_plot4.sort_values(by=['YrSold', 'MoSold']) # Sort the data by year and month
train_plot4['MoSold'] = train_plot4['MoSold'].astype(str) # Convert month to a string
train_plot4['YrSold'] = train_plot4['YrSold'].astype(str) # Convert year to a string

train_datetime = train_plot4['MoSold'] + '-' + train_plot4['YrSold'] # Concatenate the month and year strings
train_datetime = pd.DataFrame(train_datetime) # Convert to a dataframe
train_datetime.rename(columns = {0:'Col1'}, inplace = True) # Rename the first column
train_q4 = pd.DataFrame(train_datetime.Col1.unique()) # Extract only the unique row entries corresponding to each month
train_q4.rename(columns = {0:'Col1'}, inplace = True) # Rename the first column again
# Frequency tally for each month (obtained via .value_counts() function)
freq = [10,9,25,27,38,48,67,23,15,24,16,12,13,8,23,23,43,59,51,40,11,16,24,18,13,10,18,26,38,51,49,29,17,22,17,14,12,10,19,
        26,37,59,61,30,20,27,22,15,10,15,21,39,48,36,6] 

# Plot the line graph (time series plot)
train_q4['Frequency'] = freq
train_q4 = train_q4.set_index(['Col1']) # Set the date as the index for the dataframe

fig, ax = plt.subplots(figsize=(15, 8)) # Plot size
ax.plot(train_q4.index,train_q4['Frequency'],'-o',color='blue')
ax.set(xlabel="Month-Year", ylabel="Frequency",title="Frequency of Houses Sold Per Month")

# Format the x tick labels
ax.set_xticklabels(train_q4.index, rotation=90,fontsize=12)
ax.tick_params(axis = 'both', which = 'major', labelsize = 13)

plt.show() # Plot the final graph


# The number of houses sold across each year is periodic: the summer months of June and July are when the most amount of houses are sold. The lowest amount of sales occur in the winter during January and February.

# In[11]:


# Donut plot of LotShape
labels = 'Regular', 'Slightly Irregular','Moderately Irregular','Irregular' # Add the labels
values = train_10['LotShape'].value_counts() # Compute the Frequency

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)]) # Donut plot
fig.update_layout(title=go.layout.Title(text="Donut Plot of Lot Shape"))


# The donut plot reveals that 63.4% of homes have a regular lot shape while only 0.685% have a irregular lot shape. This type of plot is a better alternative to a pie chart since it is difficult to quantify size differences in a pie chart and a donut plot is more effective at showcasing individual parts.
# 
# Reference for plot: https://plot.ly/python/pie-charts/

# ## Part 3 - Handcrafted Scoring Function to Rank Homes

# #### Below the code I provide a description of the scoring function as well as list the top 10 and bottom 10 houses ranked by desirability

# In[98]:


# Select all pertinent variables for the scoring function

train_q3 = train[['OverallQual','OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual','Functional','GarageQual','GarageCond','BsmtCond',
                 'BsmtQual','PoolQC','FireplaceQu','SalePrice','YearBuilt','YearRemodAdd','LotArea','TotRmsAbvGrd',
                 'GarageCars','Alley','Fence','MiscFeature','TotalBsmtSF','GrLivArea','HalfBath',
                 'FullBath','BsmtHalfBath','BsmtFullBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF',
                 'Condition1','Condition2']]

# QUALITY PREDICTORS (see scoring function below for a description)
train_q3['OverallQual'] = train_q3['OverallQual'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4}) # Overall Quality Ranking
train_q3['OverallCond'] = train_q3['OverallCond'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4}) # Overall Condition Ranking
train_q3['ExterQual'] = train_q3['ExterQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2}) # External Quality Ranking      
train_q3['ExterCond'] = train_q3['ExterCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2}) # External Condition Ranking       
train_q3['HeatingQC'] = train_q3['HeatingQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2}) # Heating Quality Ranking      
train_q3['KitchenQual'] = train_q3['KitchenQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2}) # Kitchen Quality Ranking  
train_q3['Functional'] = train_q3['Functional'].replace({'Typ': 3,'Min1': 2, "Min2": 1, "Mod":0, "Maj1":-1, "Maj2": -2, "Sev": -3, "Sal": -4})  # Functional Quality Ranking      
train_q3['GarageQual'] = train_q3['GarageQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3}) # Garage Quality Ranking
train_q3['GarageCond'] = train_q3['GarageCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3}) # Garage Condition Ranking
train_q3['BsmtCond'] = train_q3['BsmtCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3}) # Basement Condition Ranking
train_q3['BsmtQual'] = train_q3['BsmtQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3}) # Basement Quality Ranking
train_q3['PoolQC'] = train_q3['PoolQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, np.nan: -2}) # Pool Quality Ranking
train_q3['FireplaceQu'] = train_q3['FireplaceQu'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3}) # Fireplace Quality Ranking

# HOUSE CHARACTERISTICS (see scoring function below for a description)
train_q3['NumofYearsBuilt'] = 2019-train_q3['YearBuilt'] # Age of House
train_q3['NumofYearsRemodel'] = 2019-train_q3['YearRemodAdd'] # Age of Remodel
train_q3 = train_q3.drop(['YearBuilt', 'YearRemodAdd'], axis=1)

train_q3['hasalley'] = train_q3['Alley'].apply(lambda x: 1 if pd.isnull(x) else -1) # Alley indicator
train_q3['hasfence'] = train_q3['Fence'].apply(lambda x: 1 if pd.isnull(x) else -1) # Fence indicator
train_q3['hasmisc'] = train_q3['MiscFeature'].apply(lambda x: 1 if pd.isnull(x) else -1) # Misc indicator
train_q3 = train_q3.drop(['Alley', 'Fence','MiscFeature'], axis=1)

# Union of Condition
train_q3['UnionCondition'] = (train_q3['Condition1']==train_q3['Condition2']).astype(int)
train_q3['UnionCondition'] = train_q3['UnionCondition'].replace({1: 1, 0: -1})
train_q3 = train_q3.drop(['Condition1', 'Condition2'], axis=1)

train_q3['TotalSF'] = train_q3['TotalBsmtSF'] + train_q3['GrLivArea'] # Total SF
train_q3['Total_BathSF'] = train_q3['FullBath']+ train_q3['BsmtFullBath']+ 0.5*train_q3['BsmtHalfBath']+0.5*train_q3['HalfBath'] # Total Bath SF
train_q3['Total_PorchSF'] = train_q3['OpenPorchSF']+train_q3['3SsnPorch']+train_q3['EnclosedPorch']+train_q3['ScreenPorch']+train_q3['WoodDeckSF'] # Total Porch SF
train_q3 = train_q3.drop(['FullBath','BsmtFullBath','BsmtHalfBath', 'TotalBsmtSF','GrLivArea',
                         'BsmtFullBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'], axis=1)


# In[99]:


# Standardize all scores before applying the formula
train_q3_standardized = pd.DataFrame(preprocessing.scale(train_q3))
train_q3_standardized.columns = train_q3.columns


# In[100]:


# Apply the ranking scheme: 
train_q3_standardized['ScoringResult'] = 0.50*(train_q3_standardized['OverallQual'] + train_q3_standardized['OverallCond'] + train_q3_standardized['ExterQual'] + train_q3_standardized['ExterCond'] + train_q3_standardized['HeatingQC'] +
                                train_q3_standardized['KitchenQual'] + train_q3_standardized['Functional'] + train_q3_standardized['GarageQual'] + train_q3_standardized['GarageCond'] + train_q3_standardized['BsmtCond'] +
                                train_q3_standardized['BsmtQual'] + train_q3_standardized['PoolQC'] + train_q3_standardized['FireplaceQu']) + 0.25*(train_q3_standardized['SalePrice']) + 0.25*(train_q3_standardized['NumofYearsBuilt']+
                                                                                                                                            train_q3_standardized['NumofYearsRemodel']+
                                                                                                                                            train_q3_standardized['hasalley']+
                                                                                                                                            train_q3_standardized['hasfence']+
                                                                                                                                            train_q3_standardized['hasmisc']+
                                                                                                                                            train_q3_standardized['UnionCondition']+
                                                                                                                                            train_q3_standardized['TotalSF']+
                                                                                                                                            train_q3_standardized['Total_BathSF']+
                                                                                                                                            train_q3_standardized['Total_PorchSF'])
# Extract the top 10 and the bottom 10 based on the scoring function
bottom_10 = train_q3_standardized['ScoringResult'].value_counts().sort_index().head(10)
top_10 = train_q3_standardized['ScoringResult'].value_counts().sort_index().tail(10)


# In[ ]:


bottom_10_scores = pd.DataFrame(bottom_10.index)[0] # Extract the index of the lowest scoring homes
top_10_scores = pd.DataFrame(top_10.index)[0] # Extract the index of the highest scoring homes


# In[101]:


bottom_10_row = train.loc[train_q3_standardized['ScoringResult'].isin(bottom_10_scores)] # Extract the data of the lowest scoring homes
top_10_row = train.loc[train_q3_standardized['ScoringResult'].isin(top_10_scores)] # Extract the data of the highest scoring homes


# In[102]:


# Create a boxplot showing the distribution of scores.
sns.boxplot(x = train_q3_standardized['ScoringResult']).set(xlabel='Score')
plt.title('Boxplot of Scores for my Scoring Function')
plt.show()


# The boxplot of scores for my scoring function is symmetric. There are a greater number of outliers corresponding to homes which received a low ranking score, compared to a higher score. The scores are centered around 0 since the data was standardized to account for the magnitudes of the different variables. 

# In[103]:


# Sort the bottom and top 10 scoring houses in descending order
bottom_10 = train_q3_standardized['ScoringResult'].value_counts().sort_index().head(10).sort_index(ascending=False)
top_10 = train_q3_standardized['ScoringResult'].value_counts().sort_index().tail(10).sort_index(ascending=False)

bottom_10_row = train_q3_standardized.loc[train_q3_standardized['ScoringResult'].isin(bottom_10.index)]
top_10_row = train_q3_standardized.loc[train_q3_standardized['ScoringResult'].isin(top_10.index)]

bottom_10_row = bottom_10_row.sort_values(by=['ScoringResult'],ascending=False)
top_10_row = top_10_row.sort_values(by=['ScoringResult'],ascending=False)


# In[104]:


# Note that the index column starts at 0 so the actual house id = index +1
merge_top_bot = top_10_row.append(bottom_10_row)


# 10 most desirable houses with the corresponding score
# 
# | House ID | Score |
# | --- | --- | 
# | 1183 | 17.805145 |
# | 1299 | 16.206200|
# | 198 |14.033611| 
# | 1424 | 13.571567| 
# | 584 | 10.837153 | 
# | 692 | 10.666735|
# | 186 | 10.158708|
# | 1170 | 8.837397|
# | 1244 | 8.771000|
# | 524 | 8.650593|

# 10 least desirable houses with the corresponding score
# 
# | House ID | Score |
# | --- | --- | 
# | 1219 | -11.445178 |
# | 1001 | -11.836508|
# | 40 |-12.450795 | 
# | 89 | -13.201679| 
# | 399 | -13.645170 | 
# | 251 | -15.401514|
# | 637 | -16.283081|
# | 376 | -17.816738|
# | 534 | -17.817853|
# | 706 | -18.675398|

# # Scoring Function
# 
# The scoring function will be computed as follows using a combination of rankings and weights. 
# 
# $\frac{1}{2} (\sum Quality Score) + \frac{1}{4} (\sum log(Sale Price)) + \frac{1}{4} (\sum House Characteristics) = Desirability  Score$
# 
# ## Quality Ranking
# 
# Our data set provides important information about the quality of various attributes of the house. I have converted or reassigned these categorical rankings to a numerical ranking. A negative ranking denotes poor quality houses (least desirable). Moreover, for attributes such as the Garage which has missing data, if a house does not have a garage or a pool, I consider this as negative in my desirability scoring.
# 
# All quality ranking attributes will be standardized prior to summing using the preprocessing package in sklearn. 
# 
# ##### Old Ranking: New Ranking
# 
# #### OverallQual and OverallCond
# 
#       10: 5
#       9: 4
#       8: 3
#       7: 2
#       6: 1
#       5: 0
#       4: -1
#       3: -2
#       2: -3
#       1: -4
# 
# #### ExterQual, ExterCond, HeatingQC, and KitchenQual
# 
#    - Ex (Excellent): 2
#    - Gd (Good): 1
#    - Ta (Average/Typical): 0
#    - Fa (Fair): -1
#    - Po (Poor): -2
#    
# #### Functional
#    - Typ: 3
#    - Min1: 2	
#    - Min2: 1
#    - Mod: 0
#    - Maj1: -1
#    - Maj2: -2
#    - Sev: -3
#    - Sal: -4	
# 
# #### GarageQual and GarageCond
# 
#    - Ex (Excellent): 2
#    - Gd (Good): 1
#    - Ta (Average/Typical): 0
#    - Fa (Fair): -1
#    - Po (Poor): -2
#    - NA (No Garage): -3
#    
# #### BsmtCond and BsmtQual
# 
#     - Ex (Excellent): 2
#     - Gd (Good): 1
#     - TA (Typical) - slight dampness allowed: 0
#     - Fa (Fair) - dampness or some cracking or settling: -1
#     - Po (Poor) - Severe cracking, settling, or wetness: -2
#     - NA (No Basement): -3
#     
# #### PoolQual
# 
#     - Ex (Excellent): 2
#     - Gd (Good): 1
#     - TA (Typical): 0
#     - Fa (Fair): -1
#     - NA (No Pool): -2
#     
# #### FirePlaceQual
# 
#     - Ex (Excellent): 2
#     - Gd (Good): 1
#     - TA (Typical): 0
#     - Fa (Fair): -1
#     - Po (Poor): -2
#     - NA (No Fireplace): -3
#     
# ## SalePrice
# 
# Use the log of the sale price
# 
# ## House Characteristics
# 
# #### NumofYearsBlt
# 2019 - YearBuilt
# 
# #### NumofYearsRemodel
# 2019-YearRemodAdd
# 
# #### LotArea
# 
# #### TotRmsAbvGrd
# 
# #### GarageCars
# 
# #### HasAlley
#     - Yes: 1
#     - No: -1
#     
# #### HasFence
#     - Yes: 1
#     - No: -1
#     
# #### HasMisc
#     - Yes: 1
#     - No: -1
# 
# #### Total_Square_Footage
# 
# TotalBsmtSF + GrLivArea
# 
# #### Total_Bathroom
# 
# FullBath + BsmtFullBath + 0.5 x BsmtHalfBath + 0.5 x HalfBath
# 
# #### Porch_SquareFootage
# 
# OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
# 
# #### ConditionUnion
# 
# Union of Condition 1 and Condition 2 variables: 
#     If the union size is greater > 1: assign a ranking of 1
#     If the union size is equal to 1: assign a ranking of -1

# ## Scoring Function Discussion
# 
# The individual weights and variables were incorporated into the scoring function based on the considerations of prospective homebuyers including quality, price, and spatial features. Quality predictors were weighted higher than sale price and other spatial features since it is my belief that the aesthetic quality betters captures desirability of a house even before considering its price. 
# 
# I believe my scoring function is effective. Scores with the lowest desirability level are not spacious, inexpensive, and small in size. On the other hand, houses with the highest desirability score are more spacious, expensive, and larger in size. 

# ## Part 4 - Pairwise Distance Function using Ranking Methodology

# In[105]:


# Using the scores computed in Part 3, I applied several distance metrics to
# my scoring column. Euclidean distance seemed to work best at distinguishing between 
# similar and dissimilar houses ranked by desirability. 

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import DistanceMetric
dist_m = DistanceMetric.get_metric('euclidean')
pd.DataFrame(dist_m.pairwise(pd.DataFrame(merge_top_bot['ScoringResult'])))


# The euclidean distance was used to measure the pairwise distance between each pair of houses based on the score for each home obtained from Part 3. I tried other distance functions including the chebyshev and canberra distance but the distances obtained from the latter two metrics did not conform as best to the scoring distribution in the boxplot from Part 3 of this assignment. 
# 
# Euclidean distance is effective to compare houses. The pairwise distance matrix above shows a monotonic increase in euclidean distance as houses become more dissimilar to one another (0-9 are the most desirable houses while 10-19 are the least desirable houses). Distances were very close to zero for the top and bottom scoring houses ranked by desirability than houses whose scored in the middle (close to 0) indicating its ability to differentiate between low and highly desirable properties. The distance between the highest and second highest scoring houses is 1.60 while the distance between the top and lowest scoring house is 36.48 (column 1 of the matrix). The euclidean distance measurements exhibit greater variability in comparing houses with a score near the median (0). This is one drawback to this metric.

# ## Part 5 - Clustering Homes by Neighborhood

# In[106]:


# Apply dimensionality reduction and use agglomerative clustering with euclidean distance

tsne_dim_red = TSNE(random_state=20) #Apply tsne
# tsne grid contains the new x and y coordinates for the data 
tsne_grid = tsne_dim_red.fit_transform(train_q3_standardized)

# Apply agglomerative clustering with 5 clusters
model_clust = AgglomerativeClustering(linkage='complete',affinity='euclidean', n_clusters=5)
model_clust.fit(tsne_grid)

# Create the cluster labels
labelsDF = pd.DataFrame(model_clust.labels_)
labelsDF[0] = labelsDF[0].replace({0: 'Cluster1', 1: 'Cluster5', 2:'Cluster4', 3:'Cluster2', 4: 'Cluster3'})#, 5: 'brown', 6: 'purple', 7: 'pink' , 8: 'black', 9: 'yellow'})

# Plot the figure
plt.figure(figsize=(8,8))
fig = sns.scatterplot(tsne_grid [:,0],tsne_grid [:,1],hue=list(labelsDF[0])).set_title('Agglomerative Clustering on Training Data')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(prop={'size': 14})
plt.show()


# In[107]:


# Please see below for the Neighborhood plot superimposed on the data. Note that while the OverallQual variables does not
# perfectly partition the clusters (note the overlap around the origin), it does a better job of classifying the data into
# regions than the neighborhood variable. 

plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
fig = sns.scatterplot(x=tsne_grid[:,0],y=tsne_grid[:,1],hue=train_q3_standardized['OverallQual'],s=70,palette="hot").set_title("Overall Quality")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[108]:


# Add the neighborhood variable into our data set (only the variables used in Question 3 for clustering the data above)

tsne_grid = pd.DataFrame(tsne_grid) 
tsne_grid['Neighborhood'] = train[['Neighborhood']]

plt.figure(figsize=(8,8))
fig = sns.scatterplot(tsne_grid[0],tsne_grid[1],hue=list(tsne_grid['Neighborhood']))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Neighborhood")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.show()


# First, in order to plot the clusters, I needed to apply a dimensional reduction technique in order to plot the data in two-dimensional space. I used t-distributed stochastic neighbor embedding (TSNE).
# 
# For clustering, I decided to use agglomerative clustering, a form of hierarchical clustering since this is a bottom-up approach which starts by merging each observation (house) into different clusters.
# 
# Unfortunately, the clusters do not reflect neighborhood boundaries as I hoped. The second plot shows that houses in different neighborhoods are haphazardly found in different clusters. There are few, small pockets of clusters (e.g. x = -20 and y = -20) but the plot does not separate the houses into neighborhoods and does not reflect neighborhood boundaries. 
# 
# However, when adjusting the cluster size and using an alternative variable instead of neighborhood, the clusters identified by the algorithm are similar to those exhibited by the variable "OverallQual", quantifying the overall quality of a house (see second figure). Although there is still some overlap between the overall quality of different houses near the origin of the plot, the overall quality is less for higher Y values on the plot than lower Y points which have a higher overall quality. I believe this can be attributed to my scoring function for Part 3 since quality factors (including overall quality) were weighted twice as much as sale price and spatial features of the house. 
# 
# Souce for clustering algorithms and plot: 
# 
# https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters
# 

# ## Part 6 - Baseline Linear Regression Model

# In[21]:


# I will use the following 2 independent variables to predict SalePrice, the dependent variable
# 1. TotalBsmtSF
# 2. GrLivArea

q6 = ['TotalBsmtSF','GrLivArea','SalePrice'] # Extract independent and dependent variables
train_q6 = train_10[q6] # Create a separate dataframe

y = np.log(train_q6['SalePrice']) # We will predict the log of the sale price since the distribution is skewed to the right
train_indep = train_q6.drop('SalePrice', 1)# Select the independent variables
scal_train_indep = pd.DataFrame(preprocessing.scale(train_indep)) # Scale the variables using the z-score formula
scal_train_indep = scal_train_indep.rename(columns={0: "TotalBsmtSF", 1:"GrLivArea"}) # Rename the dataframe columns

scal_train_indep = sm.add_constant(scal_train_indep) # Add an intercept to the regression model
model = sm.OLS(y, scal_train_indep).fit() ## sm.OLS(output, input)
predictions = model.predict(scal_train_indep)

# Print out the statistics
print(model.summary()) # Summmary statistics
print('-----------------------------------')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, predictions))) # Compute the RMSE


# In[22]:


# Plot the Residuals vs fitted plot
fig, ax = plt.subplots(figsize=(8, 6)) # Plot size
sns.residplot(model.fittedvalues, y, data=scal_train_indep, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
ax.set_title('Residuals vs Fitted Plot') # Title
ax.set_xlabel('Fitted values') # X-label
ax.set_ylabel('Residuals')# Y-label


# In[23]:


# Plot a Q-Q plot of the residuals
qq_plot = sm.qqplot(model.fittedvalues-y,line='s') # Q-Q Plot
qq_plot.show() # Display plot
qq_plot.suptitle('QQ Plot') # Change title


# The regression model I fit to the data was:
# $log(Sale Price) = \beta_{0} + \beta_{1}TotalBsmtSF + \beta_{2}GrLivArea$
# 
# The independent predictors (TotalBsmtSF and GrLivArea) were standardized prior to fitting the model. The adjusted coefficient of determination is 0.60, indicating that the model examines 60% of the variability in the dependent variable (log(sale price)). Both predictors are statistically significant at a 1% level of significance and the root mean square error (RMSE) of the model is low at 0.25. The Q-Q plot indicates that the residuals are normally distributed despite outliers at the right tail. The residuals vs fitted plot showed some outliers and slight curvature but overall, indicates an approprate model fit. 
# 
# Since the predictors have been standardized (subtracting the mean and dividing by its variance), the regression coefficients are standardized and may be compared with each other. These standardized coefficients quantify the mean change in log(sale price) given a one standard deviation change in the independent variable. Therefore, GrLivArea has the coefficient with the higher magnitude among the independent predictors and is a more important variable than TotalBsmtSF.
# 
# Reference for plots:
# https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034

# ## Part 7 - Adding in an External Dataset to Discern Whether Model Performance is Bolstered

# See the end of the question for a description of the datasets incorporated. 

# In[24]:


# Read in the first dataset: FTSE Nareit Mortgage Home Financing Index data

home = pd.read_csv('HomeFinancing.csv') # Read in the data set
home = home.drop([0,1,2,3,4,5], axis=0) # Delete the first 6 rows
home = home.drop(['Unnamed: 1','Unnamed: 4','Unnamed: 7','Unnamed: 9'],axis=1) # Delete the empty columns 
home = home.rename(columns={' ': 'Date', 'Unnamed: 2': 'TotalReturn','Unnamed: 3': 'TotalIndex',
                           'Unnamed: 5': 'PriceReturn','Unnamed: 6': 'PriceIndex',
                           'Unnamed: 8': 'IncomeReturn','Unnamed: 10': 'DividendYield'}) # Rename the columns

# Locate the pertinent rows for our housing dataset corresponding to the month and year the home was sold. 
#home.loc[home['Date'] == 'Jan-06'] # Located at row 79
#home.loc[home['Date'] == 'Jul-10'] # Located at row 133

home = home.loc[79:133]
home_return = home[['Date','PriceReturn']] # Return these two columns
home_return[['Date','Year']] = home_return['Date'].str.split('-',expand=True) # Concatenate these columns

# Convert the month to a number from 1-12
home_return['Date'] = home_return['Date'].replace({'Jan': 1, "Feb": 2, "Mar":3, "Apr":4, "May": 5,
                                                   'Jun': 6, "Jul": 7, "Aug":8, "Sep":9, "Oct": 10,
                                                   "Nov": 11, "Dec":12})
home_return['Year'] = home_return['Year'].replace({'06': 2006,'07': 2007,'08': 2008,'09': 2009,'10': 2010}) # Convert the year
home_return = home_return.rename(columns={'Date': 'MoSold', 'PriceReturn_Home': 'PriceReturn','Year': 'YrSold'}) # Rename cols


# In[25]:


# Read in the second dataset: FTSE Nareit Mortgage Residential Index data

res = pd.read_csv('Residential.csv') # Read in the data set
res = res.drop([0,1,2,3,4,5], axis=0) # Delete the first 6 rows
res = res.drop(['Unnamed: 1','Unnamed: 4','Unnamed: 7','Unnamed: 9'],axis=1) # Delete the empty columns 
res = res.rename(columns={' ': 'Date', 'Unnamed: 2': 'TotalReturn','Unnamed: 3': 'TotalIndex',
                           'Unnamed: 5': 'PriceReturn','Unnamed: 6': 'PriceIndex',
                           'Unnamed: 8': 'IncomeReturn','Unnamed: 10': 'DividendYield'}) # Rename the columns

# Locate the pertinent rows for our housing dataset corresponding to the month and year the home was sold. 
# res.loc[res['Date'] == 'Jan-06'] # Located at row 151
# res.loc[res['Date'] == 'Jul-10'] # Located at row 205
res = res.loc[151:205]
res_return = res[['Date','PriceReturn']] # Return these two columns

res_return[['Date','Year']] = res_return['Date'].str.split('-',expand=True)

# Convert the month to a number from 1-12
res_return['Date'] = res_return['Date'].replace({'Jan': 1, "Feb": 2, "Mar":3, "Apr":4, "May": 5,
                                                   'Jun': 6, "Jul": 7, "Aug":8, "Sep":9, "Oct": 10,
                                                   "Nov": 11, "Dec":12})
res_return['Year'] = res_return['Year'].replace({'06': 2006,'07': 2007,'08': 2008,'09': 2009,'10': 2010}) # Convert the year
res_return = res_return.rename(columns={'Date': 'MoSold', 'PriceReturn': 'PriceReturn_Res','Year': 'YrSold'}) # Rename cols


# In[26]:


# Merge with the existing dataframe 
train_home = train.merge(home_return, on=['MoSold', 'YrSold'])
train_ext_data = train_home.merge(res_return, on=['MoSold', 'YrSold'])


# The data sets I am using are monthly index values for real estate investment trusts, institutions that own a plethora of real estate investments. Residential and home price returns will be incoporated into the dataset based on the month and year that the property was sold. 
# 
# The Google drive links to the folder with the data sets. Only the price return column (column F) will be incorporated with the data.
# 
# https://drive.google.com/drive/folders/1izg-kSl6lmDxZNrz94yc7g113gGLIc6C?usp=sharing
# 
# Please see Part 9 for the code that is used to evaluate the differential accuracy of each data corpus.
# 
# #### Overview of my data science pipeline: 
# 
# 1) The training data were divided into training and validation sets encompassing 70% and 30% of the data, respectively.
# 
# 2) Missing data were imputed on a column-by-column basis. For continuous predictors (such as the GarageSF), missing data was imputed with a value of 0. For categorical predictors, missing data was imputed with the most frequent observation.
# 
# 3) Extensive feature engineering was applied (see Question 9 below). Categorical predictors were converted into dummy variables and a reference category was selected.
# 
# 4) No feature selection was applied - all predictors were included in both models.
# 
# 5) The training and validation sets were standardized by dividing by their sample mean and sample variance. 
# 
# 6) I tried using 7 different regression models (see below). I picked the method with the lowest RMSE (Gradient Boosting Regressor) and applied this method to the testing data.
# 
# The dependent variable was transformed via a log transformation. I was surprised by how poorly my model performed. These were the first two models submitted to Kaggle for scoring.

# | Method | RMSE for all variables | RMSE for all variables + external data
# | --- | --- | --- |
# | Gradient Boosting Regressor | 0.12765 |0.12773|
# | Extreme Gradient Boosting | 0.13097|0.13005|
# | Linear Regression |0.14812 | 0.14879 |
# | Lasso Regression | 0.15908| 0.15910| 
# | Support Vector Machine | 0.17212 | 0.16981|
# | Lasso with Least Angles Regression | 0.17975|0.17975
# | K-Nearest Neighbors | 0.19306| 0.19157 |
# | --- | --- | --- |
# | Kaggle Score (using Gradient Boosting Regressor) | 0.58150 | 0.58088 |

# I used gradient boosting regressor for my final model. In summary, adding these two predictors slightly decreased my Kaggle score and resulted in lower RMSE for all data mining methods, but not by much. In Part 9, I modify my data science pipeline to refine my predictions and apply feature selection. Because of these results, I decided not to incorporate my external data set into future predictions.

# ## Part 8 - Permutation Test

# In[187]:


# Permutation for single variable regression. I extract 10 independent variables from my list train subset from
# Question 1 plus the dependent variable.

train_perm = ['LotArea','LotShape','YrSold','HouseStyle','OpenPorchSF','TotalBsmtSF','GrLivArea','Neighborhood','Foundation','FullBath']
train_q8 = train[train_perm] # Select the variables for the permutation test
train_q8['YrSold'] = train_q8['YrSold'].astype(str) # Convert year to a string 

# Note the categorical predictors require that we convert them to indicator variables and
# then remove one category as the reference category
# LotShape, HouseStyle, Neighborhood, and Foundation

train_q8_log_saleprice = np.log(train['SalePrice']) # Take the log of sale price
train_q8_categorical = train_q8[['LotShape','HouseStyle','Neighborhood','Foundation','YrSold']] # Categorical
train_q8_continuous = train_q8[train_q8.columns.difference(['LotShape','HouseStyle','Neighborhood','Foundation','YrSold'])] # Continuous


# In[28]:


# Permutation Test for the categorical variables. Note that since multiple linear regression is being performed, 
# for categorical predictors, I need to convert them to dummy variables and remove one category as the reference catgeory.

p_score_list_categorical=[] # Create list of the p-values

def rmse(y_true, y_pred): # RMSE metric
     return -1.0*(np.sqrt(np.mean((y_pred-y_true))**2))

# For some reason, make_scorer multiplies the score by -1. This is why there is an extra -1 in the function above
rmse_score = make_scorer(rmse)

# First I will do the categorical variables
for i in range(len(train_q8_categorical.columns)):
    sample = np.random.randint(len(train_q8_categorical),size=100) # Select a random sample
    categorical_var = pd.get_dummies(train_q8_categorical.iloc[:,i],drop_first=True) # Create dummy variables, dropping reference category
    train_q8_log_saleprice_sample = train_q8_log_saleprice.iloc[sample.tolist()] # Log of sale price
    categorical_var_sample = categorical_var.iloc[sample.tolist()] # Extract categorical variables
    regressor = LinearRegression()   # Linear Regression
    y_pred = regressor.fit(categorical_var_sample, train_q8_log_saleprice_sample) # Fit the regression model
    score, pscore, pvalue = permutation_test_score(regressor,categorical_var_sample,train_q8_log_saleprice_sample, 
                                                   scoring=rmse_score,n_permutations=1000,cv=10,random_state=42)
    p_score_list_categorical.append(pscore)
    print("P-value : %s for %s " % (pvalue,train_q8_categorical.columns[i]))


# In[189]:


p_score_list_continuous=[] # Create list of the p-values

def rmse(y_true, y_pred): # RMSE metric
     return -1.0*(np.sqrt(np.mean((y_pred-y_true))**2))

# For some reason, make_scorer multiplies the score by -1. This is why there is an extra -1 in the function above
rmse_score = make_scorer(rmse)

# Continuous variables
for i in range(len(train_q8_continuous.columns)):
    sample = np.random.randint(len(train_q8_continuous),size=100) # Select a random sample
    continuous_var = train_q8_continuous.iloc[:,i]
    train_q8_log_saleprice_sample = train_q8_log_saleprice.iloc[sample.tolist()] # Log of sale price
    continuous_var_sample = continuous_var.iloc[sample.tolist()] # Extract categorical variables
    
    regressor = LinearRegression()   # Linear Regression
    y_pred = regressor.fit(continuous_var_sample.values.reshape(-1,1), train_q8_log_saleprice_sample) # Fit the regression model
    score, pscore, pvalue = permutation_test_score(regressor,continuous_var_sample.values.reshape(-1,1),train_q8_log_saleprice_sample, 
                                                   scoring=rmse_score,n_permutations=1000,cv=10,random_state=47)
    p_score_list_continuous.append(pscore)
    print("P-value : %s for %s " % (pvalue,train_q8_continuous.columns[i]))


# Predictor and p-values (rounded to six decimal places), sorted in descending order.
# *'s indicate that the predictor is significant at the 5% level of significance. 1,000 iterations were run using 10-fold cross-validation. The root mean square error was used as the evaluation metric.
# 
# | Independent Variable | P-Value|
# | --- | --- |
# | Foundation | 0.666334 |
# | HouseStyle | 0.311688 |
# | LotShape| 0.311688 |
# | YrSold | 0.295704 |
# | FullBath | 0.203796|
# | TotalBsmtSF | 0.103896 |
# | LotArea | 0.040959 * |
# | Neighborhood | 0.037962 * |
# | OpenPorchSF | 0.032967 * |
# | GrLivArea | 0.024975 * |

# ## Part 9 - Final Result

# ### Part 9A - Comparison of Models Run With and Without External Data Corpus

# In[30]:


# We use the training data and perform the following operations to manipulate the data:
# Feature Engineering
X_train, X_valid = train_test_split(train_ext_data,test_size=0.3, random_state=42)


# In[31]:


# Perform extensive feature engineering on the data
def FeatureEngineering(df):
    # Drop the ID column
    df = df.drop(['Id'], axis=1)
    
    # Convert MSSubClass to dummy variables, using the first category as a reference variable
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    one_hot_MSSubClass = pd.get_dummies(df['MSSubClass'],drop_first=True)
    df = df.drop('MSSubClass',axis=1)
    df = df.join(one_hot_MSSubClass)
    
    # Convert MSZoning to dummy variables, using the first category as a reference variable
    df['MSZoning'] = df['MSZoning'].replace({'A': 'A_Zone', 'C': 'C_Zone', 'I': 'I_Zone','FV': 'Res_Zone', 'RH': 'Res_Zone', 'RL': 'Res_Zone', 'RP': 'Res_Zone', 'RM': 'Res_Zone'})
    one_hot_MSZoning = pd.get_dummies(df['MSZoning'],drop_first=True)
    df = df.drop('MSZoning',axis=1)
    df = df.join(one_hot_MSZoning)
    
    # LotFrontage - convert to binary variable indicating presence of lot frontage
    df['LotFrontage'] = df['LotFrontage'].apply(lambda x: 0 if pd.isnull(x) else x)

    # LotArea: keep the same
    
    # Recategorize into binary variables
    df['Street'] = df['Street'].replace({'Grvl': 1, 'Pave': 0})
    df['Alley'] = df['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1)
    
    # LotShape: regular or irregular
    df['LotShape'] = df['LotShape'].replace({'Reg': 1, 'IR1': 0,'IR2': 0,'IR3': 0 })
    
    # LandContour: level or not level
    df['LandContour'] = df['LandContour'].replace({'Lvl': 1, 'Bnk': 0,'HLS': 0,'Low': 0 })

    # Utilities_New will quantify the number of utilities included
    df['Utilities'] = df['Utilities'].replace({'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1})
    df['Utilities_New'] = pd.Series(list(df['Utilities']), index=df.index)
    df = df.drop('Utilities',axis=1)

    # Convert LotConfig to dummy variables, using the first category as a reference variable
    one_hot_LotConfig = pd.get_dummies(df['LotConfig'],drop_first=True)
    df = df.drop('LotConfig',axis=1)
    df = df.join(one_hot_LotConfig)

    # Convert LandSlope to dummy variables, using the first category as a reference variable
    one_hot_LandSlope = pd.get_dummies(df['LandSlope'],drop_first=True)
    df = df.drop('LandSlope',axis=1)
    df = df.join(one_hot_LandSlope)
    
    # Convert Neighborhood to dummy variables, using the first category as a reference variable
    one_hot_Neighborhood = pd.get_dummies(df['Neighborhood'],drop_first=True)
    df = df.drop('Neighborhood',axis=1)
    df = df.join(one_hot_Neighborhood)

    # Union of Condition 1 and Condition 2 (indicates features nearby e.g. Railraod)
    df['UnionCondition'] = (df['Condition1']==df['Condition2']).astype(int)
    df = df.drop(['Condition1', 'Condition2'], axis=1)
    
    # Convert BldgType to dummy variables, using the first category as a reference variable
    one_hot_BldgType = pd.get_dummies(df['BldgType'],drop_first=True)
    df = df.drop('BldgType',axis=1)
    df = df.join(one_hot_BldgType)

    # Convert HousingStyle into the number of floors
    df['HouseStyle'] = df['HouseStyle'].replace({'1Story': 1, '1.5Fin': 1.5, '1.5Unf': 1.5, '2Story': 2, '2.5Fin': 2.5, '2.5Unf': 2.5, 'SFoyer': 3, 'SLvl': 3})

    # Quality and Condition Ranking: recode into a ranking scheme
    df['OverallQual'] = df['OverallQual'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4})
    df['OverallCond'] = df['OverallCond'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4})
    df['ExterQual'] = df['ExterQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2})       
    df['ExterCond'] = df['ExterCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2})       
    df['HeatingQC'] = df['HeatingQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2})       
    df['KitchenQual'] = df['KitchenQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -2})  
    df['Functional'] = df['Functional'].replace({'Typ': 3,'Min1': 2, "Min2": 1, "Mod":0, "Maj1":-1, "Maj2": -2, "Sev": -3, "Sal": -4, np.nan: -4})       
    df['GarageQual'] = df['GarageQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['GarageCond'] = df['GarageCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['BsmtCond'] = df['BsmtCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['BsmtQual'] = df['BsmtQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['PoolQC'] = df['PoolQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, np.nan: -2})
    df['FireplaceQu'] = df['FireplaceQu'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    
    # Sum the quality and condition rankings to create a variable entitled QualityPoints
    df['QualityPoints'] = df['OverallQual'] + df['OverallCond'] + df['ExterQual'] + df['ExterCond'] + df['HeatingQC'] + df['KitchenQual'] + df['Functional'] + df['GarageQual'] + df['GarageCond'] + df['BsmtCond'] + df['BsmtQual'] + df['PoolQC'] + df['FireplaceQu']
    df = df.drop(['OverallQual','OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual','Functional','GarageQual','GarageCond','BsmtCond','BsmtQual','PoolQC','FireplaceQu'],axis=1)
    df['QualityPoints_New'] = pd.Series(list(df['QualityPoints']), index=df.index)
    df = df.drop('QualityPoints',axis=1)
    
    # Calculate the age of the house
    df['AgeofHouse'] = 2019-df['YearBuilt']
    df = df.drop('YearBuilt',axis=1)

    # Calculate the remodeling age of the house
    df['AgeofRemodel'] = 2019-df['YearRemodAdd']
    df = df.drop('YearRemodAdd',axis=1)

    # Convert RoofStyle to dummy variables, using the first category as a reference variable
    one_hot_RoofStyle = pd.get_dummies(df['RoofStyle'],drop_first=True)
    df = df.drop('RoofStyle',axis=1)
    df = df.join(one_hot_RoofStyle)

    # Convert RoofMatl to dummy variables, using the first category as a reference variable
    one_hot_RoofMatl = pd.get_dummies(df['RoofMatl'],drop_first=True)
    df = df.drop('RoofMatl',axis=1)
    df = df.join(one_hot_RoofMatl)

    # Union of Exterior1st and Exterior2nd columns
    df['UnionExterior'] = (df['Exterior1st']==df['Exterior2nd']).astype(int)
    df = df.drop(['Exterior1st', 'Exterior2nd'], axis=1)

    # Convert MasVnrType to dummy variables, using the first category as a reference variable
    one_hot_MasVnrType = pd.get_dummies(df['MasVnrType'],drop_first=True)
    df = df.drop('MasVnrType',axis=1)
    df = df.join(one_hot_MasVnrType)

    # MasVnrArea: Impute with zero
    df['MasVnrArea'] = df['MasVnrArea'].apply(lambda x: 0 if pd.isnull(x) else x)

    # Convert Foundation to dummy variables, using the first category as a reference variable
    df['Foundation'] = df['Foundation'].replace({'BrkTil': 'BrkTil_Foundation', "CBlock": 'CBlock_Foundation', "PConc":'PConc_Foundation', "Slab":'Slab_Foundation', "Stone": 'Stone_Foundation', 'Wood': 'Wood_Foundation'})
    one_hot_Foundation = pd.get_dummies(df['Foundation'],drop_first=True)
    df = df.drop('Foundation',axis=1)
    df = df.join(one_hot_Foundation)
    
    # TotalBsmtSF is a sum of the following 4 variables so I delete these predictors
    df = df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtExposure'],axis=1)

    # Union of BsmtFinType1 and BsmtFinType2
    df['UnionBsmtFinType'] = (df['BsmtFinType1']==df['BsmtFinType2']).astype(int)
    df = df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1)

    # Convert Heating to dummy variables, using the first category as a reference variable
    one_hot_Heating = pd.get_dummies(df['Heating'],drop_first=True)
    df = df.drop('Heating',axis=1)
    df = df.join(one_hot_Heating)

    # Get the total square footage of the house. 1stFlrSF and 2ndFlrSF are included in the tally for GrLivArea
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df = df.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF','TotalBsmtSF','GrLivArea'], axis=1)

    # Central air: yes or no
    df['CentralAir'] = df['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)

    # Convert Electrical to dummy variables, using the first category as a reference variable
    one_hot_Electrical = pd.get_dummies(df['Electrical'],drop_first=True)
    df = df.drop('Electrical',axis=1)
    df = df.join(one_hot_Electrical)

    # Total Bathrooms
    df['Total_Bathroom'] = df['FullBath']+ 0.5*df['HalfBath']+ 0.5*df['BsmtHalfBath']+df['BsmtFullBath']
    df = df.drop(['FullBath','HalfBath','BsmtHalfBath','BsmtFullBath'], axis=1)

    # Delete other properties of the garage
    df = df.drop(['GarageType','GarageYrBlt','GarageFinish'], axis=1)
    
    # Keep GarageCars and GarageArea

    # Convert PavedDrive to dummy variables, using the first category as a reference variable
    one_hot_PavedDrive = pd.get_dummies(df['PavedDrive'],drop_first=True)
    df = df.drop('PavedDrive',axis=1)
    df = df.join(one_hot_PavedDrive)

    # Total PorchSF
    df['Total_PorchSF'] = df['OpenPorchSF']+df['3SsnPorch']+df['EnclosedPorch']+df['ScreenPorch']+df['WoodDeckSF']
    df = df.drop(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF','PoolArea'], axis=1)

    # Has fence (yes/no) and has miscellaneous feature (yes/no)
    df['Fence'] = df['Fence'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['MiscFeature'] = df['MiscFeature'].apply(lambda x: 0 if pd.isnull(x) else 1)

    # Delete MiscVal category
    df = df.drop('MiscVal', axis=1)

    # Based on my fourth plot in question 2, recode the month sold into seasons
    df['MoSold'] = df['MoSold'].replace({12:'Winter', 1: 'Winter', 2:'Winter', 3:'Spring', 4: 'Spring',
                                           5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall',
                                          10: 'Fall', 11: 'Fall', 12: 'Fall'})
    one_hot_MoSold = pd.get_dummies(df['MoSold'],drop_first=True)
    df = df.drop('MoSold',axis=1)
    df = df.join(one_hot_MoSold)

    # Convert the year sold in categorical variables
    df['YrSold'] = df['YrSold'].replace({2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4})
    pd.to_numeric(df['YrSold'])

    # Convert SaleType to dummy variables, using the first category as a reference variable
    one_hot_SaleType = pd.get_dummies(df['SaleType'],drop_first=True)
    df = df.drop('SaleType',axis=1)
    df = df.join(one_hot_SaleType)

    # Convert SaleCondition to dummy variables, using the first category as a reference variable
    one_hot_SaleCondition = pd.get_dummies(train['SaleCondition'],drop_first=True)
    df = df.drop('SaleCondition',axis=1)
    df = df.join(one_hot_SaleCondition)
    
    return df
    
X_train_FE = FeatureEngineering(X_train) # Apply feature engineering to the training data
X_train_FE['SalePrice'] = np.log(X_train_FE['SalePrice']) # Transform the dependent variable for the training data
X_valid_FE = FeatureEngineering(X_valid) # Apply feature engineering to the testing data
X_valid_FE['SalePrice'] = np.log(X_valid_FE['SalePrice']) # Transform the dependent variable for the testing data


# In[32]:


# Select the dependent variable (log of the sales price) and remove it from the independent variable datset

X_train_FE_dependent = X_train_FE['SalePrice'] # This is the log of the sales price
X_train_FE = X_train_FE.drop('SalePrice',axis=1) # Drop this from the independent variable datset

X_valid_FE_dependent = X_valid_FE['SalePrice'] # This is the log of the sales price
X_valid_FE = X_valid_FE.drop('SalePrice',axis=1) # Drop this from the independent variable datset


# In[33]:


# Standardize the data sets (subtract the predictor mean and divide by the variance)
X_train_FE_standardized = pd.DataFrame(preprocessing.scale(X_train_FE))
X_train_FE_standardized.columns = X_train_FE.columns

X_valid_FE_standardized = pd.DataFrame(preprocessing.scale(X_valid_FE))
X_valid_FE_standardized.columns = X_valid_FE.columns


# In[34]:


# Drop categorical variables that are not common to the training and validation sets
X_train_FE_standardized = X_train_FE_standardized.drop(['Wood_Foundation','GasA','Grav','CWD','ConLI','Oth','Membran','Metal','OthW'],axis=1) # Drop this from the original dataframe since we do not want to transform this variable
X_valid_FE_standardized = X_valid_FE_standardized.drop(['Mix','Con','CompShg','Roll','OthW'],axis=1) # Drop this from the original dataframe since we do not want to transform this variable


# In[35]:


# Uncomment for including the external data
X_train_FE_standardized = X_train_FE_standardized.drop(['PriceReturn_Res'],axis=1)
X_train_FE_standardized = X_train_FE_standardized.drop(['PriceReturn'],axis=1)
X_valid_FE_standardized = X_valid_FE_standardized.drop(['PriceReturn_Res'],axis=1)
X_valid_FE_standardized = X_valid_FE_standardized.drop(['PriceReturn'],axis=1)


# In[36]:


# I tried a plethora of regression algorithms to see which method yielded the lowest RMSE. 

def MachineLearning(train_independent, train_dependent, validation_independent, validation_dependent):
    
    rmse_list = [] # Store the RMSE in a list
    
    knn = KNeighborsRegressor()  # K-Nearest Neighbors
    knn.fit(pd.DataFrame(train_independent), train_dependent)
    knn_predict = knn.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-knn_predict)**2)))
    
    svm = LinearSVR(max_iter = 50000,random_state=13) # Support Vector Machine
    svm.fit(pd.DataFrame(train_independent), train_dependent)
    svm_predict = svm.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-svm_predict)**2)))

    lasso = LassoCV(random_state=13) # Lasso Regression with Cross Validation
    lasso.fit(pd.DataFrame(train_independent), train_dependent)
    lasso_predict = lasso.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-lasso_predict)**2)))
    
    gbm = GradientBoostingRegressor(random_state=13) # Gradient Boosting
    gbm.fit(pd.DataFrame(train_independent), train_dependent)
    gbm_predict = gbm.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-gbm_predict)**2)))

    xgb = XGBRegressor(random_state=13) # XG Boosting
    xgb.fit(pd.DataFrame(train_independent), train_dependent)
    xgb_predict = xgb.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-xgb_predict)**2)))

    lasso_lars = LassoLarsIC() # Lasso_Lars Regression
    lasso_lars.fit(pd.DataFrame(train_independent), train_dependent)
    lasso_lars_predict = lasso_lars.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-lasso_lars_predict)**2)))

    rf = RandomForestRegressor(random_state=13) # Random Forest Regression
    rf.fit(pd.DataFrame(train_independent),train_dependent)
    rf_predict = rf.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-rf_predict)**2)))
    
    elastic_net = ElasticNetCV(cv=10, random_state=13) # Elastic Net Regression with Cross-Validation
    elastic_net.fit(pd.DataFrame(train_independent), train_dependent)
    elastic_net_predict = elastic_net.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-elastic_net_predict)**2)))

    adaboost = AdaBoostRegressor(random_state=13) # AdaBoost Regression
    adaboost.fit(pd.DataFrame(train_independent), train_dependent)
    adaboost_predict = adaboost.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-adaboost_predict)**2)))
    
    linear_reg = LinearRegression() # Linear Regression
    linear_reg.fit(pd.DataFrame(train_independent), train_dependent)
    linear_reg_predict = linear_reg.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-linear_reg_predict)**2)))

    method_list = ['knn','svm','lasso','gbm','xgb','lasso_lars', 'linear_reg']
    
    print('Method , RMSE')
    
    for i in range(len(method_list)):
        print(method_list[i],",",round(rmse_list[i],5))
        
    return gbm

gbm_model = MachineLearning(X_train_FE_standardized, X_train_FE_dependent, X_valid_FE_standardized, X_valid_FE_dependent)


# In[37]:


# Read in the testing data and merge the external data sets with it (comment out the appropriate lines for running the model
# without the external data set)

testing = pd.read_csv('test.csv')
test_id = testing['Id']

# Merge the 2 external datasets onto the testing data set
home = pd.read_csv('HomeFinancing.csv') # Read in the data set
home = home.drop([0,1,2,3,4,5], axis=0) # Delete the first 6 rows
home = home.drop(['Unnamed: 1','Unnamed: 4','Unnamed: 7','Unnamed: 9'],axis=1) # Delete the empty columns 
home = home.rename(columns={' ': 'Date', 'Unnamed: 2': 'TotalReturn','Unnamed: 3': 'TotalIndex',
                           'Unnamed: 5': 'PriceReturn','Unnamed: 6': 'PriceIndex',
                           'Unnamed: 8': 'IncomeReturn','Unnamed: 10': 'DividendYield'}) # Rename the columns

# Locate the pertinent rows for our housing dataset corresponding to the month and year the home was sold. 
#home.loc[home['Date'] == 'Jan-06'] # Located at row 79
#home.loc[home['Date'] == 'Jul-10'] # Located at row 133

home = home.loc[79:133]
home_return = home[['Date','PriceReturn']] # Return these two columns
home_return[['Date','Year']] = home_return['Date'].str.split('-',expand=True) # Concatenate these columns

# Convert the month to a number from 1-12
home_return['Date'] = home_return['Date'].replace({'Jan': 1, "Feb": 2, "Mar":3, "Apr":4, "May": 5,
                                                   'Jun': 6, "Jul": 7, "Aug":8, "Sep":9, "Oct": 10,
                                                   "Nov": 11, "Dec":12})
home_return['Year'] = home_return['Year'].replace({'06': 2006,'07': 2007,'08': 2008,'09': 2009,'10': 2010}) # Convert the year
home_return = home_return.rename(columns={'Date': 'MoSold', 'PriceReturn_Home': 'PriceReturn','Year': 'YrSold'}) # Rename cols

res = pd.read_csv('Residential.csv') # Read in the data set
res = res.drop([0,1,2,3,4,5], axis=0) # Delete the first 6 rows
res = res.drop(['Unnamed: 1','Unnamed: 4','Unnamed: 7','Unnamed: 9'],axis=1) # Delete the empty columns 
res = res.rename(columns={' ': 'Date', 'Unnamed: 2': 'TotalReturn','Unnamed: 3': 'TotalIndex',
                           'Unnamed: 5': 'PriceReturn','Unnamed: 6': 'PriceIndex',
                           'Unnamed: 8': 'IncomeReturn','Unnamed: 10': 'DividendYield'}) # Rename the columns

# Locate the pertinent rows for our housing dataset corresponding to the month and year the home was sold. 
# res.loc[res['Date'] == 'Jan-06'] # Located at row 151
# res.loc[res['Date'] == 'Jul-10'] # Located at row 205
res = res.loc[151:205]
res_return = res[['Date','PriceReturn']] # Return these two columns

res_return[['Date','Year']] = res_return['Date'].str.split('-',expand=True)

# Convert the month to a number from 1-12
res_return['Date'] = res_return['Date'].replace({'Jan': 1, "Feb": 2, "Mar":3, "Apr":4, "May": 5,
                                                   'Jun': 6, "Jul": 7, "Aug":8, "Sep":9, "Oct": 10,
                                                   "Nov": 11, "Dec":12})
res_return['Year'] = res_return['Year'].replace({'06': 2006,'07': 2007,'08': 2008,'09': 2009,'10': 2010}) # Convert the year
res_return = res_return.rename(columns={'Date': 'MoSold', 'PriceReturn': 'PriceReturn_Res','Year': 'YrSold'}) # Rename cols

testing_home = testing.merge(home_return, on=['MoSold', 'YrSold'])
testing_ext_data = testing_home.merge(res_return, on=['MoSold', 'YrSold'])


# In[38]:


# Before feature engineering the testing data, there are some more missing entries in the data 
# not in the training and validation sets. 

# Uncomment these 2 lines for the case when we do not use the external data set
testing_ext_data = testing_ext_data.drop(['PriceReturn_Res'],axis=1) # Delete the empty columns 
testing_ext_data = testing_ext_data.drop(['PriceReturn'],axis=1) # Delete the empty columns 

# There were some additional missing entries in the training data that were not present in the training and validation sets.
# We impute them with 0 for continuous predictors and the most frequent observation for categorical columns.
testing_ext_data['Utilities'] = testing_ext_data['Utilities'].fillna('AllPub')
testing_ext_data['GarageCars'] = testing_ext_data['GarageCars'].fillna(0)
testing_ext_data['GarageArea'] = testing_ext_data['GarageArea'].fillna(0)
testing_ext_data['BsmtHalfBath'] = testing_ext_data['BsmtHalfBath'].fillna(0)
testing_ext_data['BsmtFullBath'] = testing_ext_data['BsmtFullBath'].fillna(0)
testing_ext_data['TotalBsmtSF'] = testing_ext_data['TotalBsmtSF'].fillna(0)
testing_ext_data['LotFrontage'] = testing_ext_data['LotFrontage'].fillna(0)
testing_ext_data['SaleType'] = testing_ext_data['SaleType'].fillna('WD')
testing_ext_data['MSZoning'] = testing_ext_data['MSZoning'].fillna('RL')
testing_ext_data['Exterior1st'] = testing_ext_data['Exterior1st'].fillna('VinylSd')
testing_ext_data['Exterior2nd'] = testing_ext_data['Exterior2nd'].fillna('VinylSd')
testing_ext_data['MasVnrType'] = testing_ext_data['MasVnrType'].fillna('None')

testing_FE = FeatureEngineering(testing_ext_data) # Feature Engineering
testing_FE_standardized = pd.DataFrame(preprocessing.scale(testing_FE)) # Subtract mean and divide by variance
testing_FE_standardized.columns = testing_FE.columns

testing_FE_standardized = testing_FE_standardized.drop(['150','Wood_Foundation','Grav','CWD','Con','ConLI','Oth'],axis=1) # Drop this from the original dataframe since we do not want to transform this variable

# Apply gradient boosting regressor to generate Kaggle predictions on the training set
#gbm = GradientBoostingRegressor(random_state=13) # Using 
#gbm.fit(pd.DataFrame(X_train_FE_standardized), X_train_FE_dependent)
testing_gbm_predict = gbm_model.predict(testing_FE_standardized)


# In[39]:


np.exp(testing_gbm_predict)
test_id_DF = pd.DataFrame(test_id)
final_df = test_id_DF.join(pd.DataFrame(np.exp(testing_gbm_predict)))
#final_df.to_csv('HW3_Kaggle_5.csv',index=False)


# ### Part 9B: Improved Predictions for Kaggle Submission

# To improve my previous Kaggle scores, I modified my data science pipeline
# 
# 1) The training data were divided into training and validation sets encompassing 70% and 30% of the data, respectively.
# 
# 2) Missing data were evaluated on a column-by-column basis. For continuous predictors (such as the GarageSF), missing data was imputed with a value of 0. For categorical predictors, missing data was imputed with the most frequent observation.
# 
# 3) Extensive feature engineering was applied (see Question 9 below). Categorical predictors were converted into dummy variables and a reference category was selected. I performed additional feature engineering which included combining additional categorical factors such as the materials used for the house.
# 
# 4) The training and validation sets were standardized by dividing by their sample mean and sample variance. 
# 
# 5) Feature selection using XGBoost was applied to remove insignificant predictors from the model to reduce overfitting. The top three predictors that were identified in my training data as being the most significant predictors were (1) QualityPoints (this was my custom aggregation of the quality variables -- see Question 3 for a more detailed description), (2) LotArea, and (tied for 3) Age of House and Total SF. This yielded 49 predictors and it helped to substantially improve my Kaggle score.
# 
# 6) I tried using 11 different regression models (see below). The top two methods with the lowest RMSE were applied to the testing data (Gradient Boosting and XGB).
# 
# 
# Three additional entries were submitted to Kaggle:
# 
# 1) Gradient Boosting - Kaggle score: 0.14687 
# 
# 2) Extreme Gradient Boosting - Kaggle score: 0.14243
# 
# 3) Linear combination of Gradient Boosting (GBM), Support Vector Machine (SVM), and Lasso Regression (Lasso): 
# 
# 0.6 x XGBoost + 0.2 x SVM + 0.2 x Lasso - Kaggle score: 0.13656 *** Best Score
# 
# The linear combination of machine learning methods dramatically increased my Kaggle score. 

# In[40]:


# Read in the training data
train = pd.read_csv('train.csv')
X_train, X_valid = train_test_split(train,test_size=0.3, random_state=42)


# In[41]:


# Extensive feature engineering

def FeatureEngineering(df):
    df = df.drop(['Id'], axis=1) # Drop the Id column

    # Convert MSSubClass to dummy variables, using the first category as a reference variable
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    one_hot_MSSubClass = pd.get_dummies(df['MSSubClass'],drop_first=True)
    df = df.drop('MSSubClass',axis=1)
    df = df.join(one_hot_MSSubClass)

    # Convert MSZoning to dummy variables, using the first category as a reference variable
    df['MSZoning'] = df['MSZoning'].replace({'A': 'A_Zone', 'C': 'C_Zone', 'I': 'I_Zone','FV': 'Res_Zone', 'RH': 'Res_Zone', 'RL': 'Res_Zone', 'RP': 'Res_Zone', 'RM': 'Res_Zone'})
    one_hot_MSZoning = pd.get_dummies(df['MSZoning'],drop_first=True)
    df = df.drop('MSZoning',axis=1)
    df = df.join(one_hot_MSZoning)

    # LotFrontage- fill missing entries in with 0
    df['LotFrontage'] = df['LotFrontage'].apply(lambda x: 0 if pd.isnull(x) else x)

    # LotArea: keep the same

    df['Street'] = df['Street'].replace({'Grvl': 1, 'Pave': 0}) # Gravel or Paved street
    df['Alley'] = df['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1) # Alley indicator
    df['LotShape'] = df['LotShape'].replace({'Reg': 1, 'IR1': 0,'IR2': 0,'IR3': 0 }) # LotShape: regular or irregular
    df['LandContour'] = df['LandContour'].replace({'Lvl': 1, 'Bnk': 0,'HLS': 0,'Low': 0 }) # LandContour: level or not-level

    # Utilities will now quantify the number of utilities that are included
    df['Utilities'] = df['Utilities'].replace({'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1})
    df['Utilities_New'] = pd.Series(list(df['Utilities']), index=df.index)
    df = df.drop('Utilities',axis=1)

    # Convert LotConfig to dummy variables, using the first category as a reference variable
    one_hot_LotConfig = pd.get_dummies(df['LotConfig'],drop_first=True)
    df = df.drop('LotConfig',axis=1)
    df = df.join(one_hot_LotConfig)

    # Convert LandSlope to dummy variables, using the first category as a reference variable
    one_hot_LandSlope = pd.get_dummies(df['LandSlope'],drop_first=True)
    df = df.drop('LandSlope',axis=1)
    df = df.join(one_hot_LandSlope)
    
    # Convert Neighborhood to dummy variables, using the first category as a reference variable
    one_hot_Neighborhood = pd.get_dummies(df['Neighborhood'],drop_first=True)
    df = df.drop('Neighborhood',axis=1)
    df = df.join(one_hot_Neighborhood)

    # Condition 1 and Condition 2 Union
    df['UnionCondition'] = (df['Condition1']==df['Condition2']).astype(int)
    df = df.drop(['Condition1', 'Condition2'], axis=1)
    
    # Convert BldgType to dummy variables, using the first category as a reference variable
    one_hot_BldgType = pd.get_dummies(df['BldgType'],drop_first=True)
    df = df.drop('BldgType',axis=1)
    df = df.join(one_hot_BldgType)

    # HousingStyle: use number of stories in the house
    df['HouseStyle'] = df['HouseStyle'].replace({'1Story': 1, '1.5Fin': 1.5, '1.5Unf': 1.5, '2Story': 2, '2.5Fin': 2.5, '2.5Unf': 2.5, 'SFoyer': 3, 'SLvl': 3})

    # Quality Points
    df['OverallQual'] = df['OverallQual'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4,np.nan: -5})
    df['OverallCond'] = df['OverallCond'].replace({10: 5, 9: 4, 8: 3, 7: 2, 6: 1, 5: 0, 4: -1, 3: -2, 2: -3, 1: -4,np.nan: -5})
    df['ExterQual'] = df['ExterQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2,np.nan: -3})       
    df['ExterCond'] = df['ExterCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2,np.nan: -3})       
    df['HeatingQC'] = df['HeatingQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2,np.nan: -3})       
    df['KitchenQual'] = df['KitchenQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2,np.nan: -3})  
    df['Functional'] = df['Functional'].replace({'Typ': 3,'Min1': 2, "Min2": 1, "Mod":0, "Maj1":-1, "Maj2": -2, "Sev": -3, "Sal": -4,np.nan: -5})       
    df['GarageQual'] = df['GarageQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['GarageCond'] = df['GarageCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['BsmtCond'] = df['BsmtCond'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['BsmtQual'] = df['BsmtQual'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})
    df['PoolQC'] = df['PoolQC'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, np.nan: -2})
    df['FireplaceQu'] = df['FireplaceQu'].replace({'Ex': 2, "Gd": 1, "TA":0, "Fa":-1, "Po": -2, np.nan: -3})

    df['QualityPoints'] = df['OverallQual'] + df['OverallCond'] + df['ExterQual'] + df['ExterCond'] + df['HeatingQC'] + df['KitchenQual'] + df['Functional'] + df['GarageQual'] + df['GarageCond'] + df['BsmtCond'] + df['BsmtQual'] + df['PoolQC'] + df['FireplaceQu']
    df = df.drop(['OverallQual','OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual','Functional','GarageQual','GarageCond','BsmtCond','BsmtQual','PoolQC','FireplaceQu'],axis=1)
    df['QualityPoints_New'] = pd.Series(list(df['QualityPoints']), index=df.index)
    df = df.drop('QualityPoints',axis=1)

    df['AgeofHouse'] = 2019-df['YearBuilt'] # Age of the house 
    df = df.drop('YearBuilt',axis=1)

    df['AgeofRemodel'] = 2019-df['YearRemodAdd'] # Age of remodeling
    df = df.drop('YearRemodAdd',axis=1)

    # Convert RoofStyleto dummy variables, using the first category as a reference variable
    one_hot_RoofStyle = pd.get_dummies(df['RoofStyle'],drop_first=True)
    df = df.drop('RoofStyle',axis=1)
    df = df.join(one_hot_RoofStyle)

    # Convert RoofMatl to dummy variables, using the first category as a reference variable
    one_hot_RoofMatl = pd.get_dummies(df['RoofMatl'],drop_first=True)
    df = df.drop('RoofMatl',axis=1)
    df = df.join(one_hot_RoofMatl)

    # Union of Exterior1st and Exterior2nd
    df['UnionExterior'] = (df['Exterior1st']==df['Exterior2nd']).astype(int)
    df = df.drop(['Exterior1st', 'Exterior2nd'], axis=1)

    # Convert MasVnrType to dummy variables, using the first category as a reference variable
    one_hot_MasVnrType = pd.get_dummies(df['MasVnrType'],drop_first=True)
    df = df.drop('MasVnrType',axis=1)
    df = df.join(one_hot_MasVnrType)

    # MasVnrArea: Impute with zero for missing entries
    df['MasVnrArea'] = df['MasVnrArea'].apply(lambda x: 0 if pd.isnull(x) else x)

    # Convert Foundation to dummy variables, using the first category as a reference variable
    df['Foundation'] = df['Foundation'].replace({'BrkTil': 'BrkTil_Foundation', "CBlock": 'CBlock_Foundation', "PConc":'PConc_Foundation', "Slab":'Slab_Foundation', "Stone": 'Stone_Foundation', 'Wood': 'Wood_Foundation'})
    one_hot_Foundation = pd.get_dummies(df['Foundation'],drop_first=True)
    df = df.drop('Foundation',axis=1)
    df = df.join(one_hot_Foundation)

    df = df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtExposure'],axis=1)

    # Union of BsmtFinType
    df['UnionBsmtFinType'] = (df['BsmtFinType1']==df['BsmtFinType2']).astype(int)
    df = df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1)

    # Convert Heating to dummy variables, using the first category as a reference variable
    one_hot_Heating = pd.get_dummies(df['Heating'],drop_first=True)
    df = df.drop('Heating',axis=1)
    df = df.join(one_hot_Heating)

    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea'] # Total Square Footage
    df = df.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], axis=1)

    df['CentralAir'] = df['CentralAir'].apply(lambda x: 1 if x=='Y' else 0) # Central Air - binary variable

    # Convert Electrical to dummy variables, using the first category as a reference variable
    one_hot_Electrical = pd.get_dummies(df['Electrical'],drop_first=True)
    df = df.drop('Electrical',axis=1)
    df = df.join(one_hot_Electrical)

    # Total Bathrooms
    df['Total_Bathroom'] = df['FullBath']+ 0.5*df['HalfBath']+ 0.5*df['BsmtHalfBath']+df['BsmtFullBath']
    df = df.drop(['FullBath','HalfBath','BsmtHalfBath','BsmtFullBath'], axis=1)

    # Delete other properties of the garage
    df = df.drop(['GarageType','GarageYrBlt','GarageFinish'], axis=1)
    # Keep GarageCars and GarageArea

    # Convert PavedDrive to dummy variables, using the first category as a reference variable
    one_hot_PavedDrive = pd.get_dummies(df['PavedDrive'],drop_first=True)
    df = df.drop('PavedDrive',axis=1)
    df = df.join(one_hot_PavedDrive)

    # PorchSF
    df['Total_PorchSF'] = df['OpenPorchSF']+df['3SsnPorch']+df['EnclosedPorch']+df['ScreenPorch']+df['WoodDeckSF']
    df = df.drop(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF','PoolArea'], axis=1)

    df['Fence'] = df['Fence'].apply(lambda x: 0 if pd.isnull(x) else 1) # Fence or no fence
    df['MiscFeature'] = df['MiscFeature'].apply(lambda x: 0 if pd.isnull(x) else 1) # Misc feature or not
    df = df.drop('MiscVal', axis=1)

    # Month the house was sold 
    df['MoSold'] = df['MoSold'].replace({12:'Winter', 1: 'Winter', 2:'Winter', 3:'Spring', 4: 'Spring',
                                           5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall',
                                          10: 'Fall', 11: 'Fall', 12: 'Fall'})
    one_hot_MoSold = pd.get_dummies(df['MoSold'],drop_first=True)
    df = df.drop('MoSold',axis=1)
    df = df.join(one_hot_MoSold)

    # Year the house was sold
    df['YrSold'] = df['YrSold'].replace({2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4})
    pd.to_numeric(df['YrSold'])

    # Convert SaleType to dummy variables, using the first category as a reference variable
    one_hot_SaleType = pd.get_dummies(df['SaleType'],drop_first=True)
    df = df.drop('SaleType',axis=1)
    df = df.join(one_hot_SaleType)

    # Convert SaleCondition to dummy variables, using the first category as a reference variable
    one_hot_SaleCondition = pd.get_dummies(df['SaleCondition'],drop_first=True)
    df = df.drop('SaleCondition',axis=1)
    df = df.join(one_hot_SaleCondition)
    
    return df
    
X_train_FE = FeatureEngineering(X_train) # Apply feature engineering to the training data
X_train_FE['SalePrice'] = np.log(X_train_FE['SalePrice']) # Log of sale price will be used as the dependent variable
X_valid_FE = FeatureEngineering(X_valid) # Apply feature engineering to the validation data
X_valid_FE['SalePrice'] = np.log(X_valid_FE['SalePrice']) # Log of sale price will be used as the depedent variable


# In[42]:


X_train_FE_dependent = X_train_FE['SalePrice'] # This is the log of the sales price
X_train_FE = X_train_FE.drop('SalePrice',axis=1) # Drop this from the original dataframe since we do not want to transform this variable

X_valid_FE_dependent = X_valid_FE['SalePrice'] # This is the log of the sales price
X_valid_FE = X_valid_FE.drop('SalePrice',axis=1) # Drop this from the original dataframe since we do not want to transform this variable


# In[43]:


# Standardize the data sets for training and validation sets 
X_train_FE_standardized = pd.DataFrame(preprocessing.scale(X_train_FE))
X_train_FE_standardized.columns = X_train_FE.columns

X_valid_FE_standardized = pd.DataFrame(preprocessing.scale(X_valid_FE))
X_valid_FE_standardized.columns = X_valid_FE.columns


# In[44]:


# Apply Feature Selection only to the training data. XGB worked the best in HW 2 so I will use 
# recursive feature elimination to perform the task.

xgb = XGBRegressor()
xgb.fit(X_train_FE_standardized, X_train_FE_dependent)
imp = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_train_FE_standardized.columns)
imp = imp.sort_values(['Importance'], ascending = False)

# Variable Importance plot for the top 15 features
plot_importance(xgb,max_num_features=15)
pyplot.show()


# In[45]:


# Define a function to calculate RMSE

def neg_rmse(y_true, y_pred):
    return -1.0*(np.sqrt(np.mean((y_true-y_pred)**2)))

neg_rmse = make_scorer(neg_rmse) # RMSE - make_scorer makes the values negative by default so I multiply by -1

# Apply Recursive Feature Elimination with 5-fold cross-validation
estimator = XGBRegressor()
selector = RFECV(estimator, cv = 5, n_jobs = -1, scoring = neg_rmse)
selector = selector.fit(X_train_FE_standardized, X_train_FE_dependent)

# print("The number of selected features is: {}".format(selector.n_features_))

# Print the predictors that will be utilized in the final model. 
features_kept = X_train_FE_standardized.columns.values[selector.support_] 
print("Features kept: {}".format(features_kept))


# In[46]:


# Subset the data to only extract the significant predictors
train_FS = X_train_FE_standardized[['LotFrontage','LotArea','LandContour','HouseStyle','MasVnrArea','TotalBsmtSF','CentralAir','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','MiscFeature','YrSold' ,'30' ,'60' ,'70', 'Res_Zone' ,'CulDSac' ,'Inside', 'BrkSide' ,'ClearCr','Crawfor', 'Edwards' ,'MeadowV', 'NAmes', 'OldTown', 'Somerst' ,'StoneBr','Veenker', 'UnionCondition' ,'QualityPoints_New' ,'AgeofHouse','AgeofRemodel', 'Gable' ,'CBlock_Foundation' ,'UnionBsmtFinType', 'Grav','TotalSF', 'Total_Bathroom', 'Total_PorchSF', 'Spring' ,'Winter' ,'New','Alloca' ,'Family', 'Normal']]
valid_FS = X_valid_FE_standardized[['LotFrontage','LotArea','LandContour','HouseStyle','MasVnrArea','TotalBsmtSF','CentralAir','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','MiscFeature','YrSold' ,'30' ,'60' ,'70', 'Res_Zone' ,'CulDSac' ,'Inside', 'BrkSide' ,'ClearCr','Crawfor', 'Edwards' ,'MeadowV', 'NAmes', 'OldTown', 'Somerst' ,'StoneBr','Veenker', 'UnionCondition' ,'QualityPoints_New' ,'AgeofHouse','AgeofRemodel', 'Gable' ,'CBlock_Foundation' ,'UnionBsmtFinType', 'Grav','TotalSF', 'Total_Bathroom', 'Total_PorchSF', 'Spring' ,'Winter' ,'New','Alloca' ,'Family', 'Normal']]


# In[47]:


# Machine Learning function

def MachineLearning(train_independent, train_dependent, validation_independent, validation_dependent):
    rmse_list = [] # Return all the RMSE in a list
    
    knn = KNeighborsRegressor() # K-Nearest Neighbors
    knn.fit(pd.DataFrame(train_independent), train_dependent)
    knn_predict = knn.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-knn_predict)**2)))
    
    svm = LinearSVR(max_iter = 50000,random_state=13) # Linear Support Vector Regression
    svm.fit(pd.DataFrame(train_independent), train_dependent)
    svm_predict = svm.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-svm_predict)**2)))

    sgd = SGDRegressor(random_state=13) # Stochastic Gradient Descent Regression
    sgd.fit(pd.DataFrame(train_independent), train_dependent)
    sgd_predict = sgd.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-sgd_predict)**2)))

    lasso = LassoCV(random_state=13) # Lasso Regression with Cross-Validation
    lasso.fit(pd.DataFrame(train_independent), train_dependent)
    lasso_predict = lasso.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-lasso_predict)**2)))
    
    gbm = GradientBoostingRegressor(random_state=13) # Gradient Boosting
    gbm.fit(pd.DataFrame(train_independent), train_dependent)
    gbm_predict = gbm.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-gbm_predict)**2)))

    xgb = XGBRegressor(random_state=13) # Extreme Gradient Boosting
    xgb.fit(pd.DataFrame(train_independent), train_dependent)
    xgb_predict = xgb.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-xgb_predict)**2)))

    lasso_lars = LassoLarsIC() # Lasso model fit with Lars
    lasso_lars.fit(pd.DataFrame(train_independent), train_dependent)
    lasso_lars_predict = lasso_lars.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-lasso_lars_predict)**2)))

    rf = RandomForestRegressor(random_state=13) # Random Forest
    rf.fit(pd.DataFrame(train_independent),train_dependent)
    rf_predict = rf.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-rf_predict)**2)))
    
    elastic_net = ElasticNetCV(cv=10, random_state=13) # Elastic Net Regression with 10-fold cross-validation
    elastic_net.fit(pd.DataFrame(train_independent), train_dependent)
    elastic_net_predict = elastic_net.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-elastic_net_predict)**2)))

    adaboost = AdaBoostRegressor(random_state=13) # AdaBoost
    adaboost.fit(pd.DataFrame(train_independent), train_dependent)
    adaboost_predict = adaboost.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-adaboost_predict)**2)))
    
    linear_reg = LinearRegression() # Multiple Linear Regression
    linear_reg.fit(pd.DataFrame(train_independent), train_dependent)
    linear_reg_predict = linear_reg.predict(validation_independent)
    rmse_list.append(np.sqrt(np.mean((validation_dependent-linear_reg_predict)**2)))

    method_list = ['knn','svm','sgs','lasso','gbm','xgb','lasso_lars', 'linear_reg']
    
#     Uncomment to print out selected methods/output

#     print('Method , RMSE')
    
#     for i in range(len(method_list)):
#         print(method_list[i],",",round(rmse_list[i],5))
        
    return xgb
    # return [xgb,svm,lasso]

gbm_model = MachineLearning(train_FS, X_train_FE_dependent, valid_FS, X_valid_FE_dependent)
# [xgb_model,svm_model,lasso_model] = MachineLearning(train_FS, X_train_FE_dependent, valid_FS, X_valid_FE_dependent)


# In[48]:


# read in the testing data
test = pd.read_csv('test.csv')
test_id = test['Id']


# In[49]:


# There were more columns that were missing entries in the testing data than training data. For categorical predictors
# I imputed entries with the most frequent category. For only continuous variables, I imputed the missing values with 0. In this
# context using 0 is appropriate (e.g. if the house does not have a garage, the garage area is zero)

test['Utilities'] = test['Utilities'].fillna('AllPub')
test['GarageCars'] = test['GarageCars'].fillna(0)
test['GarageArea'] = test['GarageArea'].fillna(0)
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['LotFrontage'] = test['LotFrontage'].fillna(0)
test['SaleType'] = test['SaleType'].fillna('WD')
test['MSZoning'] = test['MSZoning'].fillna('RL')
test['Exterior1st'] = test['Exterior1st'].fillna('VinylSd')
test['Exterior2nd'] = test['Exterior2nd'].fillna('VinylSd')
test['MasVnrType'] = test['MasVnrType'].fillna('None')

testing_FE = FeatureEngineering(test) # Feature Engineering
testing_FE_standardized = pd.DataFrame(preprocessing.scale(testing_FE)) # Subtract mean and divide by variance
testing_FE_standardized.columns = testing_FE.columns


# In[50]:


# Extract the significant predictors that were used to build the model on the training data
test_FS = testing_FE_standardized[['LotFrontage','LotArea','LandContour','HouseStyle','MasVnrArea','TotalBsmtSF','CentralAir','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','MiscFeature','YrSold' ,'30' ,'60' ,'70', 'Res_Zone' ,'CulDSac' ,'Inside', 'BrkSide' ,'ClearCr','Crawfor', 'Edwards' ,'MeadowV', 'NAmes', 'OldTown', 'Somerst' ,'StoneBr','Veenker', 'UnionCondition' ,'QualityPoints_New' ,'AgeofHouse','AgeofRemodel', 'Gable' ,'CBlock_Foundation' ,'UnionBsmtFinType', 'Grav','TotalSF', 'Total_Bathroom', 'Total_PorchSF', 'Spring' ,'Winter' ,'New','Alloca' ,'Family', 'Normal']]


# In[51]:


# Generate the predictions
testing_gbm_predict = gbm_model.predict(test_FS)
# testing_gbm_predict = 0.6*xgb_model.predict(test_FS)+0.2*svm_model.predict(test_FS)+0.2*lasso_model.predicy(test_FS)


# In[52]:


# Convert the dependent variable back to its original scaling and make Kaggle submissions. 
test_id_DF = pd.DataFrame(test_id)
final_df = test_id_DF.join(pd.DataFrame(np.exp(testing_gbm_predict)))
#final_df.to_csv('HW3_Kaggle_4Submission.csv',index=False)


# # Part 10: Kaggle Results

# Score: 0.13656

# Number of entries: 6
