import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

myList=[]
d = open("auto-mpg.data","r")
#print (re.findall(r'"(.*?)"', d.readline()))
#lines = d.readline().split('   ')
for line in d: #parse file
    if '?' in line:
        continue
    temp = re.split(r'\s+', line)
    for i in range(0,8):
        temp[i] = float(temp[i])
    #print (temp)
    
    myList.append(temp)

        


df= pd.DataFrame(myList,columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year', 'origin','car name','','','','','',''])

Xc = []
Yc = []
polyRes = []
fig, ax = plt.subplots(figsize=(8,8))
feature = 'displacement'#change to which ever plot you want to see

def poly(category, order): #creating the coefficients, category takes in the feature above and order is the order you would like to plot
    global Xc
    global Yc
    global polyRes
    xRow=[]
    column = df[category].head(300)
    yVec = df['mpg'].head(300)
    y = np.matrix(yVec)
    y2 = np.transpose(y)
    
    order = order + 1
    
    for i in range(len(column)):
        RowData = []
        for x in range(0,order):
            RowData.append((float(column[i]))**(x))
            
        xRow.append(RowData)
    
    matrix = np.matrix(xRow)
    
    
    test = np.dot(np.linalg.inv(np.dot(np.transpose(matrix),matrix)), np.dot(np.transpose(matrix), y2 ))#Creating coefficient matrix
    polyRes.append(test)#store coeff matrix in polyRes
    
    xVals = np.linspace(np.min(column) - 1, np.amax(column) + 1, 500)
    
    
    yArry = [0] * len(xVals)#plotting the coeffcients
    for k in range(len(test)):
        yArry += test[k] * (xVals ** k)
    plt.xlabel(category)
    plt.ylabel('mpg')
    Yc = np.transpose(yArry)
    Xc = np.array(xVals)
    


def errorP(category):#find MSE
    
    act = df['mpg'].head(300)
    sAct = 0
    
    for j in range(len(act)):
        sAct += act[j]
    data = df[category].head(300)
    sError = 0
    predict = 0.0
    
    for i in range(len(data)):
        for index, coeff in enumerate(polyRes[0]): #CHANGE to polyRes[0,1,2,3,4] for the 4,3,2,1,0 order respectively
            predict += (coeff * (data[i]**index))
        #print(predict) # "final" predicted mpg
        # e = (actual mpg - predicted mpg)^2 ... sError += e
        error = (act[i] - predict)**2
        sError += error
        predict = 0
        #print('-----')    
        
    sError /= 300
    #print(sError)
    print(sError, 'this is error')
    


poly(feature,4)

ax.plot(Xc, Yc,color='blue')
poly(feature,3)

ax.plot(Xc, Yc,color='red')
poly(feature,2)
ax.plot(Xc, Yc,color='yellow')
poly(feature,1)

ax.plot(Xc, Yc, color = 'green')
poly(feature,0)
ax.plot(Xc, Yc, color='black')


plt.scatter(df[feature].head(300),df['mpg'].head(300))

plt.show()

errorP(feature)

#bParam = 

    
    