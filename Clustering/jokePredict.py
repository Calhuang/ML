import sqlite3
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

def cat_encoding(series):
    # code strings into integers to feed into model
    encoder = preprocessing.LabelEncoder()
    encoder.fit(series)
    column = encoder.transform(series)
    return column


db = sqlite3.connect('red_team.db')
cursor = db.cursor()

cursor.execute('''SELECT * FROM JokeRater''')
all_rows = cursor.fetchall()
#print(all_rows)
cursor.execute('''SELECT * FROM Joke''')
all_rows2 = cursor.fetchall()
cursor.execute('''SELECT * FROM JokeRating''')
all_rows3 =  cursor.fetchall()

user_feat = pd.DataFrame(all_rows)
joke_feat = pd.DataFrame(all_rows2)
joke_rate = pd.DataFrame(all_rows3)




#print(joke_rate[joke_rate.columns[2]],joke_feat[joke_feat.columns[0]])
#print(joke_feat.loc[joke_feat[joke_feat.columns[0]]==joke_rate[joke_rate.columns[2]]])
#user_joke_feat = pd.concat([user_joke_feat,user_joke_feat],axis=1)

#print(user_joke_feat.loc[num_user_ratings] = pd.concat([user_joke_feat.loc[num_user_ratings],joke_feat.loc[joke_feat[0]==joke_rate[2][0]]],axis=1))
def build_user_joke_feat():
    user_joke_feat = pd.DataFrame()
    for num_user_ratings in range(0,len(joke_rate)):
        #print(user_joke_feat.loc[num_user_ratings] = pd.concat([user_joke_feat.loc[num_user_ratings],joke_feat.loc[joke_feat[0]==joke_rate[2][num_user_ratings]]],axis=1))
        user_joke_feat = user_joke_feat.append(joke_feat.loc[joke_feat[0]==joke_rate[2][num_user_ratings]])
        #user_joke_feat.loc[num_user_ratings] = user_joke_feat.loc[num_user_ratings].append(joke_feat.loc[joke_feat[0]==joke_rate[2][num_user_ratings]])
        
    return(user_joke_feat)
    
#build_out = build_user_joke_feat()  #comment to stop building everytime

#frames = [joke_rate, build_out]
#user_joke_feat = pd.concat(frames,axis=1)

build_out = build_out.reset_index(drop=True)    

user_joke_feat = joke_rate.join(build_out,lsuffix='joke_rate', rsuffix='build_out')

clean_ujf = user_joke_feat.drop(user_joke_feat.columns[[0,1,4,9,10]],axis=1)

