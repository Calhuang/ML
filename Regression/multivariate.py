import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

myList=[]
d = open("auto-mpg.data","r")
#print (re.findall(r'"(.*?)"', d.readline()))
#lines = d.readline().split('   ')
for line in d:
    if '?' in line:
        continue
    temp = re.split(r'\s+', line)
    for i in range(0,8):
        temp[i] = float(temp[i])
    #print (temp)
    
    myList.append(temp)

        


df= pd.DataFrame(myList,columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year', 'origin','car name','1','2','3','4','5','6'])
dfback = df
df2 = df
df = df.drop('mpg',1)
df = df.drop('1',1)
df = df.drop('2',1)
df = df.drop('3',1)
df = df.drop('4',1)
df = df.drop('5',1)
df = df.drop('6',1)
df = df.drop('car name',1)



df.insert(0,'one',1)

df2 = df2.drop('mpg',1)
df2 = df2.drop('1',1)
df2 = df2.drop('2',1)
df2 = df2.drop('3',1)
df2 = df2.drop('4',1)
df2 = df2.drop('5',1)
df2 = df2.drop('6',1)
df2 = df2.drop('car name',1)



df2.insert(0,'one',1)


df['cylinders2'] = df['cylinders']**2
df['displacement2'] = df['displacement']**2
df['horsepower2'] = df['horsepower']**2
df['weight2'] = df['weight']**2
df['acceleration2'] = df['acceleration']**2
df['model year2'] = df['model year']**2
df['origin2'] = df['origin']**2
#df = second order df2= first order

df3 = df2['one']
matrixM = np.array(df3.head(300))
matrixM = np.matrix(matrixM)
matrixM = np.transpose(matrixM)
print(matrixM)
polyRes=[]


def poly(): #generate coeffeicents from first 300

    global polyRes
    global matrixM
    
    yVec = dfback['mpg'].head(300)
    y = np.matrix(yVec)
    y2 = np.transpose(y)
    
    
    #test = np.linalg.inv((np.dot(np.transpose(matrix),matrix)))
    #test2 = np.dot(np.transpose(matrix),y2)
    #print(np.dot(np.transpose(matrixM),matrixM))
    test = np.dot(np.linalg.inv(np.dot(np.transpose(matrixM),matrixM)), np.dot(np.transpose(matrixM), y2 ))
    polyRes.append(test)
    



def errorP():#calc MSE for test
    global matrixM
    global df
    global df2
    act = dfback.tail(92)['mpg']
    act = np.array(act)
    
    X = df3.tail(92)
    #print("hi", np.array(X.iloc[0]))

    sError = 0
    coefficients = np.array(polyRes[0])
    #print("coefficients", coefficients)
    
    for i in range(len(X.index)):
        row = np.array(X.iloc[i])
        result = np.dot(row, coefficients)
        #print(result)

        err = (act[i] - result)**2

        sError += err
        #print('-----')    
        
    sError /= 92
    #print(sError)
    
    print(sError, 'this is error test')
    
def errorP2(): #calc MSE for train
    global matrixM
    global df
    global df2
    act2 = dfback.head(300)['mpg']
    act2 = np.array(act2)
    
    X = df3.head(300)
    #print("hi", np.array(X.iloc[0]))

    sError2 = 0
    coefficients2 = np.array(polyRes[0])
    #print("coefficients", coefficients2)
    
    for i in range(len(X.index)):
        row2 = np.array(X.iloc[i])
        result2 = np.dot(row2, coefficients2)
        #print(result2)

        err2 = (act2[i] - result2)**2

        sError2 += err2
       # print('-----')    
        
    sError2 /= 300
    #print(sError)
    
    print(sError2, 'this is error train')
    


poly()


errorP()
errorP2()

#bParam = 


