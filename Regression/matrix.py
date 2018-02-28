
import pandas as pd
import re
from pandas.plotting import scatter_matrix

myList=[]
d = open("auto-mpg.data","r")
#print (re.findall(r'"(.*?)"', d.readline()))
#lines = d.readline().split('   ')
for line in d:
    temp = re.split(r'\s+', d.readline())
    for i in range(0,8):
        temp[i] = float(temp[i])
    #print (temp)
    if "?" not in temp:
        myList.append(temp)
    #print(myList)
    
df= pd.DataFrame(myList,columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year', 'origin','car name','','','','',''])
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')




#df = pd.DataFrame((my_2darray), columns=['a', 'b', 'c', 'd'])

#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')