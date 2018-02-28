import sqlite3
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#HOW TO RUN:
    #load the matrix:
    #user_matrix = load_data()
    
    #run the error testing:
    #results = test(user_matrix,7) // THIS WILL TAKE A LONG TIME
    
    
    #convert data to fit into function:
    #user_matrix = user_matrix.astype(int)
    
    #get the error matrix:
    #error_results = computeErrors(user_matrix,results)

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
    user_vector = np.delete(user_vector, [0,1,9,10], axis=1)  
    user_vector = np.array(user_vector)
    
    df = pd.DataFrame(all_rows3, columns=['id', 'rating', 'jokeId', 'raterId'])
    df2 = df[df.raterId.isin([all_rows[0][0]])]
    # =============================================================================
    #Debugging objects
    # df3 = df2.sort_values('jokeId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # df4 = df.sort_values('raterId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # df5 = df.sort_values(['raterId', 'jokeId'], axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # =============================================================================
    
    user_vector.resize(117, 170) #may have to change this if dataset grows
    user_vector[:, 7:170] = 0 #change the zero to whatever placeholder for gaps
    

    
    for i in range(0,len(all_rows)):
        df2 = df[df.raterId.isin([all_rows[i][0]])]
        for j in range(0,(len(df2))):
            index = df2.iloc[j]['jokeId'] - 505
            #print(df2.iloc[j]['jokeId'])
            user_vector[i][index+7] = df2.iloc[j]['rating']
    
    
    for i in range(0,7):
        user_vector[:,i] = cat_encoding(user_vector[:,i])
    
    return(user_vector)
        


    
    
def predict_joke(input_vector, user_matrix, num_user_features, numClusters=5):
    # holds clustered data
    clusterMatrix = pd.DataFrame(user_matrix).astype(int)
    # holds joke ratings
    r = [i for i in range(0,num_user_features)]
    rating_matrix = clusterMatrix.drop(r, axis = 1)

    # holds indeces of columns that input_vector does not have info for
    drop_list = []
    
    # the array used to predict the cluster from input_vector
    input_features = input_vector[0:num_user_features]
    
    #compute drop_list and append to input_feature
    for i in range(num_user_features, len(input_vector)):
        if input_vector[i] == 0:
            drop_list.append(i)
        else:
            input_features += [input_vector[i]]
            
    # make cluster matrix only hold data that input vector has
    clusterMatrix = clusterMatrix.drop(drop_list, axis = 1)
    
    # perform KMeans clustering
    kmeans = KMeans(n_clusters = numClusters).fit(clusterMatrix)
    
    # predict which cluster input_features belongs to
    cluster = kmeans.predict([input_features])
   
    # find indices of cluster
    cluster_index = np.equal(kmeans.labels_, cluster)
    # select joke ratings from cluster
    cluster_rating_matrix = rating_matrix[cluster_index]
    
    # get unseen jokes list from drop_list
    unseen_jokes = np.subtract(drop_list,num_user_features)
    return vote_joke(cluster_rating_matrix, unseen_jokes)




def vote_joke(rating_matrix, unseen_jokes):
    voteAccumulator = [0 for i in range(0,len(rating_matrix.iloc[0]))]
    numVotes = [1 for i in range (0, len(rating_matrix.iloc[0]))]
    for row in rating_matrix.as_matrix():
        for j in unseen_jokes:
            if row[j] != 0:
                voteAccumulator[j] += row[j]
                numVotes[j] += 1
             
    avgRatings = np.divide(voteAccumulator,numVotes)
    
    # return -1 if ran out of jokes (all joke ratings are zero)
    if max(avgRatings) == 0:
        return -1

    return np.argmax(avgRatings)
            
            


def test(user_matrix, num_user_features, numClusters = 5):
    user_matrix = pd.DataFrame(user_matrix)
    numRows = len(user_matrix)
    numCols = len(user_matrix.iloc[0])
    outputRatingOrders = []
    for i in range (0,numRows):
        data_index = [j for j in range(0,numRows) if j != i]
        train_data = user_matrix.iloc[data_index]
        real_input_vector = user_matrix.iloc[i].astype(int).tolist()
        test_input_vector = real_input_vector[0:num_user_features]
        test_input_vector += [0 for j in range(0, numCols - num_user_features)]
        # while predict function returns new joke
        outputRatingOrder = []
        print (i)
        while True:
            joke = predict_joke(test_input_vector, train_data,
                                num_user_features, numClusters)
            if joke >= 0:
                jokeIndex = joke + num_user_features
                jokeRating = real_input_vector[jokeIndex]
                outputRatingOrder += [jokeRating]
                if jokeRating > 0:
                    test_input_vector[jokeIndex] = jokeRating
                # need to make sure 0 isn't fed back into system
                else:
                    test_input_vector[jokeIndex] = -1
                
            else:
                break
        outputRatingOrders += [outputRatingOrder]
        
    return outputRatingOrders


def computeError(ratingsList, outputRatingOrder):
    idealRatingOrder = sorted(ratingsList)
    idealRatingOrder.reverse()
    idealAccum = 0;
    outputAccum = 0;
    errorAccum = 0;
    randomErrorAccum = 0;
    averageRating = sum(idealRatingOrder) / sum(np.not_equal(idealRatingOrder, 0))
    idealIndex = 0
    outputIndex = 0
    tpr = 0
    fpr = 0
    tpr2 = 0
    fpr2 = 0
    count = 0
    while (idealIndex < len(idealRatingOrder)) & (outputIndex < len(outputRatingOrder)):
        idealVal = idealRatingOrder[idealIndex]
        outputVal = outputRatingOrder[outputIndex]
        #print(idealVal)
        #print(outputVal)
        # skip outputVal of 0 because that means joke wasn't rated
        if outputVal == 0 :
            outputIndex += 1
            continue;
        idealAccum += idealVal
        outputAccum+= outputVal
        errorAccum += idealAccum - outputAccum
        if (idealAccum - outputAccum) > (.1*idealAccum):
            tpr += 1    
        else:
            fpr += 0
        count += 1
        randomErrorAccum += idealAccum - idealIndex * averageRating
        if (idealAccum - idealIndex * averageRating) > (.1*idealAccum):
            tpr2 += 1    
        else:
            fpr2 += 0
        idealIndex += 1
        outputIndex += 1
        
    if (idealIndex == 0):
        return [0,0,0,0,0]

    

    
    return [errorAccum/randomErrorAccum, errorAccum / idealIndex, errorAccum, idealIndex, randomErrorAccum / idealIndex, randomErrorAccum]

        
        
        
def computeErrors(data, outputRatingOrders, usrFeatures = 7):
    data = pd.DataFrame(data)
    outputErrors = []
    outputErrors2 = list()

    outputIndex = 0
    plt.show()
    for row in data.as_matrix():
        outputErrors += [computeError(row[usrFeatures:], outputRatingOrders[outputIndex])]
        outputErrors2.append( [computeError(row[usrFeatures:], outputRatingOrders[outputIndex])])
        outputIndex += 1
        
        #plt.plot(roc_fpr,roc_tpr,lw=2,alpha=1, color='b')

    return outputErrors

        
    






