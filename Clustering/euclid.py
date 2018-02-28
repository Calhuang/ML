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

def load_data():
    db = sqlite3.connect('red_team.db')
    cursor = db.cursor()
    
    cursor.execute('''SELECT * FROM JokeRater''')
    all_rows = cursor.fetchall()
    #print(all_rows)
    cursor.execute('''SELECT * FROM Joke''')
    all_rows2 = cursor.fetchall()
    cursor.execute('''SELECT * FROM JokeRating''')
    all_rows3 =  cursor.fetchall()
    
    user_vector = np.matrix(all_rows)
    user_vector = np.delete(user_vector, [0,1], axis=1)  
    user_vector = np.array(user_vector)
    
    df = pd.DataFrame(all_rows3, columns=['id', 'rating', 'jokeId', 'raterId'])
    df2 = df[df.raterId.isin([all_rows[0][0]])]
    # =============================================================================
    #Debugging objects
    # df3 = df2.sort_values('jokeId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # df4 = df.sort_values('raterId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # df5 = df.sort_values(['raterId', 'jokeId'], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # =============================================================================
    
    user_vector.resize(117, 172) #may have to change this if dataset grows
    user_vector[:, 9:172] = 0 #change the zero to whatever placeholder for gaps
    

    
    for i in range(0,len(all_rows)):
        df2 = df[df.raterId.isin([all_rows[i][0]])]
        for j in range(0,(len(df2))):
            index = df2.iloc[j]['jokeId'] - 505
            #print(df2.iloc[j]['jokeId'])
            user_vector[i][index+9] = df2.iloc[j]['rating']
    
    
    for i in range(0,9):
        user_vector[:,i] = cat_encoding(user_vector[:,i])
    
    return(user_vector)
        


    
    
def predict_joke(input_vector, user_matrix, num_user_features):
    clusterMatrix = pd.DataFrame(user_matrix)
    user_matrix = pd.DataFrame(user_matrix)
    drop_list = []
    input_features = input_vector[0:num_user_features]
    for i in range(num_user_features, len(input_vector)):
        if input_vector[i] == 0:
            drop_list.append(i)
        else:
            input_features = input_features.append(input_vector[i])
            
            
    clusterMatrix = clusterMatrix.drop(drop_list, axis = 1)
    kmeans = KMeans(n_clusters = 5).fit(clusterMatrix)
    cluster = kmeans.predict([input_features])
   
    cluster_index = np.equal(kmeans.labels_, cluster)
    cluster_matrix = user_matrix[cluster_index]

    return cluster_matrix