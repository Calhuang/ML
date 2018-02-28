import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import re
from keras.callbacks import LambdaCallback
from keras.callbacks import History 
import matplotlib.pyplot as plt

train_num = 1484
test_num = 445
epoch_num = 60
diff = train_num - test_num

myList=[]
d = open("yeast.data","r")
#print (re.findall(r'"(.*?)"', d.readline()))
def populate_y (labels,array,choose): #creates the y array for fittting + populates it with the right classes
    if choose == 'train':
        for i in range(0,train_num):
            if(labels[i]=='CYT'):
                array[i][0] = 1
            elif(labels[i]=='NUC'):
                array[i][1] = 1
            elif(labels[i]=='MIT'):
                array[i][2] = 1
            elif(labels[i]=='ME3'):
                array[i][3] = 1
            elif(labels[i]=='ME2'):
                array[i][4] = 1
            elif(labels[i]=='ME1'):
                array[i][5] = 1
            elif(labels[i]=='EXC'):
                array[i][6] = 1
            elif(labels[i]=='VAC'):
                array[i][7] = 1
            elif(labels[i]=='POX'):
                array[i][8] = 1
            elif(labels[i]=='ERL'):
                array[i][9] = 1
            else:
                return 0
    elif choose == 'test':
        for i in range(0,test_num):
            if(labels[i+diff]=='CYT'):
                array[i][0] = 1
            elif(labels[i+diff]=='NUC'):
                array[i][1] = 1
            elif(labels[i+diff]=='MIT'):
                array[i][2] = 1
            elif(labels[i+diff]=='ME3'):
                array[i][3] = 1
            elif(labels[i+diff]=='ME2'):
                array[i][4] = 1
            elif(labels[i+diff]=='ME1'):
                array[i][5] = 1
            elif(labels[i+diff]=='EXC'):
                array[i][6] = 1
            elif(labels[i+diff]=='VAC'):
                array[i][7] = 1
            elif(labels[i+diff]=='POX'):
                array[i][8] = 1
            elif(labels[i+diff]=='ERL'):
                array[i][9] = 1
            else:
                return 0
  
for line in d: #importing data...
    if '?' in line:
        continue
    temp = re.split(r'\s+', line)
    for i in range(1,9):
        temp[i] = float(temp[i])

    
    myList.append(temp)



original_data = np.array(myList)
np.random.shuffle(original_data) #randomize data
labels = original_data[:,9]
del_data = np.delete(original_data, [0,9,10], axis=1)
data = del_data.astype(float)

a = np.zeros(shape=(train_num,10)) #training set for y
a2 = np.zeros(shape=(test_num,10)) #testing set for y

populate_y(labels,a,'train') 
populate_y(labels,a2,'test') 

x_train = data[:train_num, :]
y_train = a
x_test = data[:test_num, :]
y_test = a2

model = Sequential() #keras model for back prop using SGD

model.add(Dense(3, activation='sigmoid', input_shape=(8,)))

model.add(Dropout(0.5))


model.add(Dense(10, activation='sigmoid'))
weight_list = []
loss_list = []
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: weight_list.append(model.layers[2].get_weights()))
test_loss =  LambdaCallback(on_epoch_end=lambda batch, logs: loss_list.append(model.evaluate(x_test, y_test, batch_size=128)))
history = History()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=epoch_num,
          batch_size=128,callbacks = [print_weights, history,test_loss])
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
test_sample =np.matrix([0.58, 0.47, 0.54, 0.11, 0.50, 0.00, 0.51, 0.26])
prediction = model.predict(test_sample)
print('Prediction:', prediction)


weight1 = [] #first weight for CYT
weight2 = [] #second weight for CYT
weight3 = [] #third weight for CYT
weight4 = [] #bias weight for CYT
weight5 = [] #loss for test set

for i in range(0,epoch_num):
    weight1.append(weight_list[i][0][0][0])
    weight2.append(weight_list[i][0][1][0])
    weight3.append(weight_list[i][0][2][0])
    weight4.append(weight_list[i][1][0])
    weight5.append(loss_list[i][0])
    
y = range(epoch_num)
line1, = plt.plot(y,weight1,label='first weight')
line2, = plt.plot(y,weight2,label='second weight')
line3, = plt.plot(y,weight3,label='third weight')
line4, = plt.plot(y,weight4,label='bias weight')

plt.legend(handles=[line1,line2,line3,line4])
plt.ylabel('weights for CYT')
plt.xlabel('number of epochs')
plt.axis([0, epoch_num,-1,1])

plt.show()

line1, = plt.plot(y,history.history['loss'],label='train loss')
line2, = plt.plot(y,weight5,label='test loss')
plt.legend(handles=[line1,line2])
plt.ylabel('loss for CYT')
plt.xlabel('number of epochs')
plt.show()




                
    
    