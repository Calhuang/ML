import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[2]:

rater_df = pd.read_csv("JokeRater.csv")
rating_df = pd.read_csv("JokeRating.csv")
joke_df = pd.read_csv("Joke.csv")


# Looking at the CSVs.

# In[9]:

rater_df.head()


# In[3]:

rater_df.rename(columns = {'id':'joke_rater_id'}, inplace = True)
joke_df.rename(columns = {'id':'joke_id'}, inplace = True)
joke_df['joke_id'] = joke_df['joke_id'].astype(float)
rater_df = rater_df.drop('joke_submitter_id', axis=1)
joke_df = joke_df.drop('joke_submitter_id', axis=1)
joke_df = joke_df.drop('joke_source', axis=1)


rating_df.head()



joke_df.head()



df = pd.merge(rating_df, rater_df, on="joke_rater_id", how="outer")
df = df.drop('id', axis=1)

df = pd.merge(df, joke_df, on='joke_id', how='outer')
df.head()



for i in range(14):
    df.iloc[:,i] = df.iloc[:,i].fillna('0')




df.rating.unique() # all the rating types in the database




def convert_one_hot(series):
    # code strings into integers to feed into model
    encoder = preprocessing.LabelEncoder()
    encoder.fit(series)
    column = encoder.transform(series)
    return column

for i in [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    df.iloc[:,i] = convert_one_hot(df.iloc[:,i])


