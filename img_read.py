import os
import torch
import random
from PIL import Image
from torchvision.transforms import ToTensor
import pickle
from tqdm import tqdm

DATADIR = input("please enter the file name you have image data in:  ")

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



IMG_SIZE = 224


i=1
list_id = []
label = {}
split = 0.8

for classes in main_classes:
    print(i)
    i = i+1
    path1 =os.path.join(DATADIR,classes)
    
    for category in CATEGORIES :
        path = os.path.join(path1, category)
        
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
             	if img.endswith('.jpg'):

                    f_path = os.path.join(path, img)

                    img_array = Image.open(f_path)

                    img_array = ToTensor()(img_array)

                    file_loc = os.path.join(path, img[:-4])

                    torch.save(img_array, file_loc+'.pt')
                    
                    list_id.append(file_loc+'.pt')

                    label[file_loc+'.pt'] = class_num
                    
                    #label.append(class_num)

            except Exception as e:
                pass            



partition = {'train': list_id[0:round(len(list_id)*split)],'test': list_id[round(len(list_id)*split):]}
pickle.dump(list_id,open(os.path.join(DATADIR,'list_id.p'), 'wb'))
pickle.dump(label, open(os.path.join(DATADIR,'label.p'), 'wb'))
pickle.dump(partition, open(os.path.join(DATADIR,'partition.p'), 'wb'))

print(">>done<<")