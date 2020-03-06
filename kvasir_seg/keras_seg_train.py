import numpy as np
import sys, pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score

#https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import bce_jaccard_loss

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


def preprocess_input(X): return X/127.5 - 1 
def preprocess_output(Y): return np.asarray(Y > 0.1 * np.max(Y), dtype=np.uint8)    


model_id = int(sys.argv[1]) #1: UNet, 2: Res-UNet, 3: LinkNet, 4: Res-LinkNet, 5: FPN, 6: Res-FPN, 7: PSPNet, 8: Res-PSPNet


# import   
if model_id in [7, 8]: image_dim = 384
else: image_dim = 320

with open('kvasir_seg_'+str(image_dim)+'.pickle', 'rb') as f:
    [X, Y] = pickle.load(f)
    print(X.shape, Y.shape)

Y = np.expand_dims(Y, 3)

assert np.max(X)==255
assert np.min(X)==0
assert np.max(Y)==1
assert np.min(Y)==0


# data split
X_trnval, X_tst, Y_trnval, Y_tst = train_test_split(X, Y, test_size=100, random_state=27407)
X_trn, X_val, Y_trn, Y_val = train_test_split(X_trnval, Y_trnval, test_size=100, random_state=27407)

# model construction (keras + sm)
if model_id == 1: model = sm.Unet(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 2: model = sm.Unet('resnet50', classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 3: model = sm.Linknet(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 4: model = sm.Linknet('resnet50', classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 5: model = sm.FPN(classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 6: model = sm.FPN('resnet50', classes=1, activation='sigmoid', encoder_weights='imagenet')
elif model_id == 7: model = sm.PSPNet(classes=1, activation='sigmoid', encoder_weights='imagenet') # input size must be 384x384
elif model_id == 8: model = sm.PSPNet('resnet50', classes=1, activation='sigmoid', encoder_weights='imagenet') # input size must be 384x384


# data augmentation and scaling
data_gen_args = dict(rotation_range=360, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.15, brightness_range=[0.5, 1.5], horizontal_flip=True, vertical_flip=False, fill_mode='constant', cval=0)

image_generator = ImageDataGenerator(**data_gen_args, preprocessing_function = preprocess_input)
mask_generator = ImageDataGenerator(**data_gen_args, preprocessing_function = preprocess_output)

batch_size = 10

image_flow = image_generator.flow(X_trn, seed = 27407, batch_size=batch_size)
mask_flow = mask_generator.flow(Y_trn, seed = 27407, batch_size=batch_size)
data_flow = zip(image_flow, mask_flow)

tmpbatchx = image_flow.next()
tmpbatchy = mask_flow.next()
assert np.max(tmpbatchx)==1
assert np.min(tmpbatchx)==-1
assert np.max(tmpbatchy)==1
assert np.min(tmpbatchy)==0

X_val = preprocess_input(X_val)
assert np.max(X_val)==1
assert np.min(X_val)==-1

X_tst = preprocess_input(X_tst)
assert np.max(X_tst)==1
assert np.min(X_tst)==-1

print(tmpbatchx.shape, tmpbatchy.shape)
print(X_tst.shape, Y_tst.shape)

    
# training
print(':: training')
model.compile(optimizer=Adam(lr=1e-4), loss=bce_jaccard_loss, metrics=[IOUScore(threshold=0.5, per_image=True)]) #loss='binary_crossentropy'
#model.summary()

earlystopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1)
mcp_save = ModelCheckpoint('mnnet_seg_'+str(model_id)+'.h5', save_best_only=True, monitor='val_iou_score', mode='max')

model.fit_generator(data_flow, steps_per_epoch=np.ceil(len(X_trn)/batch_size), epochs=500, validation_data=(X_val, Y_val), callbacks=[earlystopper, reduce_lr, mcp_save], verbose=2)

# prediction
print(':: prediction')
model = load_model('mnnet_seg_'+str(model_id)+'.h5', custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss, 'iou_score': IOUScore(threshold=0.5, per_image=True)})

Y_tst_hat = (model.predict(X_tst, batch_size=batch_size) > 0.5) + 0
print('mIoU:', np.mean([jaccard_score(Y_tst[i].ravel(), Y_tst_hat[i].ravel()) for i in range(len(Y_tst))]))
#pickle.dump( [np.array(127.5*(X_tst[:10]+1), dtype=np.uint8), np.array(Y_tst[:10], dtype=np.uint8), np.array(Y_tst_hat[:10], dtype=np.uint8)], open( "tstresult.p", "wb" ) )
