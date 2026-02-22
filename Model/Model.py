import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import gc
import caer
import canaro

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.legacy import SGD


IMG_SIZE = (80, 80)
channels = 1

data_dir = r'../input/the-simpsons-characters-dataset/simpsons_dataset'


# ------------------------------------------------
#---------- getting top characters----------------
# ------------------------------------------------

charCount = {}

for folder in os.listdir(data_dir):
    path = os.path.join(data_dir, folder)
    if os.path.isdir(path):
        charCount[folder] = len(os.listdir(path))

charCount = caer.sort_dict(charCount, descending=True)

characters = []
for i in range(10):
    characters.append(charCount[i][0])

print("Using characters:", characters)


# --------------------------------------------
# -------------------loading data-------------
# --------------------------------------------

data = caer.preprocess_from_dir(
    data_dir,
    characters,
    channels=channels,
    IMG_SIZE=IMG_SIZE,
    isShuffle=True
)

print("Total samples:", len(data))


# just checking one image
plt.imshow(data[0][0], cmap='gray')
plt.show()


# separate features and labels
X, y = caer.sep_train(data, IMG_SIZE=IMG_SIZE)

X = caer.normalize(X)
y = to_categorical(y, len(characters))


# train / val split
x_train, x_val, y_train, y_val = caer.train_val_split(X, y, val_ratio=0.2)

# free memory
del data
del X
del y
gc.collect()


BATCH_SIZE = 32
EPOCHS = 10


# data generator (helps accuracy a bit)
gen = canaro.generators.imageDataGenerator()
train_gen = gen.flow(x_train, y_train, batch_size=BATCH_SIZE)


# -------------------------------------
# -----------building model------------
# -------------------------------------

w, h = IMG_SIZE

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 input_shape=(w, h, channels)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

# output layer
model.add(Dense(len(characters), activation='softmax'))

model.summary()


# -----------------------------------------
# ----------------training-----------------
# -----------------------------------------

opt = SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

callbacks = [LearningRateScheduler(canaro.lr_schedule)]

history = model.fit(
    train_gen,
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=len(y_val)//BATCH_SIZE,
    callbacks=callbacks
)

print("Classes:", characters)


# ------------------------------------
# ----------testing single image------
# ------------------------------------

test_img_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'

img = cv.imread(test_img_path)

plt.imshow(img)
plt.show()


def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img


pred = model.predict(prepare(img))
print("Prediction:", characters[np.argmax(pred[0])])
