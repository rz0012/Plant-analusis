import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

file_list = []
class_list = []

DATADIR = "C:\Ram\work\plant analysis\Annotated iNaturalist Dataset pre-pt\Annotated iNaturalist Dataset"

CATEGORIES = ["branches", "leaves", "trees"]

main_classes = ["abiesconcolor","acercampestre","acerginnala","acernegundo","acerpalmatum","acerpensylvanicum","acerpseudoplatanus","acersaccharum",
            "aesculusglabra","aesculushippocastamon","aesculuspavi","ailanthusaltissima","albiziajulibrissin","amelanchierarborea","amelanchierlaevis",
            "asiminatriloba","betulalenta","betulanigra","carpinuscaroliniana","caryacordiformis","caryaglabra","catalpabignonioides","chionanthusvirginicus",
            "cornusmas","crataeguscrus-galli","cryptomeriajaponica","diospyrosvirginiana","ficuscarica","fraxinusamericana","fraxinuspennsylvanica",
            "ginkgobiloba","gleditsiatriacanthos","gymnocladusdioicus","ilexopaca","juglanscinerea","juglansnigra","koelreuteriapaniculata","liquidambarstyraciflua",
            "liriodendrontulipifera","maclurapomifera","magnoliaacuminata","magnoliagrandiflora","magnoliamacrophylla","magnoliatripetala","magnoliavirginiana",
            "maluscoronaria","maluspumila","morusalba","nyssasylvatica","ostryavirginiana","oxydendrumarboreum","piceapungens","pinusflexilis","pinusstrobus",
            "platanusoccidentalis","populusdeltoides","prunuspensylvanica","prunusserotina","pyruscalleryana","quercusacutissima","quercusbicolor","quercuscoccinea",
            "quercusimbricaria","quercusmichauxii","quercusmuehlenbergii","quercusphellos","quercusrobur","quercusshumardii","quercusvelutina",
            "salixcaroliniana","staphyleatrifolia","taxodiumdistichum","tiliacordata"]


IMG_SIZE = 64


training_data = []
i=1
for classes in main_classes:
    print(i)
    i = i+1
    path1 =os.path.join(DATADIR,classes)
    
    for category in CATEGORIES :
        path = os.path.join(path1, category)
        
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE,3))
                training_data.append([new_array, class_num])
                
            except Exception as e:
                pass




random.shuffle(training_data)



X = [] 
y = [] 

for features, label in training_data:
	X.append(features)
	y.append(label)



X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 3).astype('float32')


y = to_categorical(y)



X = X/255.0

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


#this is a basic convolutional neural network implemetation, which is commented out 
#because are not using it roght now. 
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(3))
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit(X, y, batch_size=50, epochs=15, validation_split=0.2, verbose = 1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc,label="accuracy")
plt.legend()
plt.plot(epochs, val_acc, label="val_accuracy")
plt.legend()
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, label="loss")
plt.legend()
plt.plot(epochs, val_loss,label="val_loss")
plt.legend()
plt.title('Training and validation loss')
plt.show()



model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')
'''
