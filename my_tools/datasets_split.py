ration=0.9 #train:trainval

src_image_dir="/Users/jt/Desktop/UISEE/image/Raw_Images/"
src_annotation_dir="/Users/jt/Desktop/UISEE/image/detection_labels/"

train_set_file="/Users/jt/Desktop/UISEE/image//train.txt"
trainval_set_file="/Users/jt/Desktop/UISEE/image//trainval.txt"
#test_set_file="test.txt"

import os
import random
all_annotations=os.listdir(src_annotation_dir)
random.shuffle(all_annotations)

train_f=open(train_set_file,'w')
trainval_f=open(trainval_set_file,'w')
thosld=len(all_annotations)*ration
train_k=0
for anno in all_annotations:
	if anno[-3:]=='txt' and os.path.isfile(os.path.join(src_image_dir,anno[:-3]+"png")):
		if train_k<thosld:
			train_f.writelines(anno[:-4]+'\n')
			train_k+=1
		else:
			trainval_f.writelines(anno[:-4]+'\n')
	else:
		print "file:",anno,"has no image"
train_f.close()
trainval_f.close()
