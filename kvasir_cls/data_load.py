import numpy as np
import sys, os, pickle
import cv2
import matplotlib.pyplot as plt

from preprocessing import text_vanish as tv
from preprocessing import crop_edges as ce


image_dim = int(sys.argv[1])#224, 299, 331, ...

dpath_original = './datav2'
dpath_preprocess = dpath_original+'_preprocess'
class_list = ['esophagitis', 'dyed-lifted-polyps', 'dyed-resection-margins', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

# preprocessing
if not(os.path.isdir(dpath_preprocess)):
    os.mkdir(dpath_preprocess)

    for i, c in enumerate(class_list):
    
        print(c)
        
        class_dir = os.path.join(dpath_original, c)
        file_list = os.listdir(class_dir)
        
        out_dir = os.path.join(dpath_preprocess, c)
        if not(os.path.isdir(out_dir)): os.mkdir(out_dir) 
        
        for f in file_list:
        
            impath = os.path.join(class_dir, f)
            # attempt to remove text from image (if present)
            processed = tv.Vanisher(impath).do_vanish()
            # crop black edges from image
            cropped = ce.Crop(processed).do_crop()
            # final image is resized to one uniform size
            final = cv2.resize(cropped, (625, 532))
    
            # difine filename with suffix and write to output directory
            filename = os.path.basename(impath).strip()[:-4] + '_prsd'
            out_path = os.path.join(out_dir, filename + '.png')
            cv2.imwrite(out_path, final)


X=[]
Y=[]
for i, c in enumerate(class_list):

    class_dir = os.path.join(dpath_preprocess, c)
    file_list = os.listdir(class_dir)
    
    for f in file_list:
    
        impath = os.path.join(class_dir, f)
        
        x = cv2.imread(impath)        
        x = np.round(cv2.resize(x, (image_dim, image_dim)))
        x.dtype=np.uint8
        X.append(x)
        Y.append([i])

        if len(X) % 100 == 0:
            print(len(X), 'done')

X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

print(X.shape, Y.shape)
print(X.dtype, Y.dtype)
print(np.max(X), np.min(X))

with open('kvasir_cls_'+str(image_dim)+'.pickle', 'wb') as f:
    pickle.dump([X, Y], f)