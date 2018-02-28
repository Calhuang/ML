from keras.layers import Dense
from keras.models import Sequential
import re
import numpy as np
import keras

myList=[]
d = open("yeast.data","r")
#print (re.findall(r'"(.*?)"', d.readline()))
#lines = d.readline().split('   ')
for line in d:
    if '?' in line:
        continue
    temp = re.split(r'\s+', line)
    for i in range(1,9):
        temp[i] = float(temp[i])
    #print (temp)
    
    myList.append(temp)

#print(myList)

original_data = np.array(myList)
del_data = np.delete(original_data, [0,9,10], axis=1)
data = del_data.astype(float)
print(data)

labels = np.random.randint(8, size=(1484, 1))

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)