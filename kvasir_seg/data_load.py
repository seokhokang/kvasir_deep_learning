import numpy as np
import sys, os, skimage, pickle
import cv2
import matplotlib.pyplot as plt


image_dim = int(sys.argv[1]) #320, 384, ...

dpath = './data'

X=[]
Y=[]

image_list = os.listdir(os.path.join(dpath, 'images'))
mask_list = os.listdir(os.path.join(dpath, 'masks'))

for f in image_list:

    image_path = os.path.join(dpath, 'images', f)
    mask_path = os.path.join(dpath, 'masks', f)

    x = cv2.imread(image_path)  
    x = np.round(cv2.resize(x, (image_dim, image_dim)))
    x.dtype=np.uint8
    
    y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_dim, image_dim))

    y = (y > 0.1 * np.max(y)) + 0

    X.append(x)
    Y.append(y)

    if len(X) % 100 == 0:
        print(len(X), 'done')

X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

print(X.shape, Y.shape)
print(X.dtype, Y.dtype)
print(np.max(X), np.min(X))

with open('kvasir_seg_'+str(image_dim)+'.pickle', 'wb') as f:
    pickle.dump([X, Y], f)