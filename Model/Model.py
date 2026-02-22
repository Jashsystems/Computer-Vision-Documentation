import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

IMG_SIZE = 80
DATA_PATH = r'C:\Users\User\Desktop\simpsons_dataset'  # CHANGE THIS

# Get top 10 characters
char_count = {}

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if os.path.isdir(folder_path):
        char_count[folder] = len(os.listdir(folder_path))

# Sort and take top 10
characters = sorted(char_count, key=char_count.get, reverse=True)[:10]

print("Characters:", characters)

# Load images
X = []
y = []

for idx, character in enumerate(characters):
    path = os.path.join(DATA_PATH, character)
    for img_name in os.listdir(path):
        try:
            img = cv.imread(os.path.join(path, img_name))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)
        except:
            continue

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y, len(characters))

# Train validation split
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(characters), activation='softmax'))

optimizer = SGD(learning_rate=0.001, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)

# Test example
test_image_path = r'C:\Users\User\Desktop\test.jpg'  # CHANGE THIS

img = cv.imread(test_image_path)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_resized = cv.resize(img_gray, (IMG_SIZE, IMG_SIZE))
img_input = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

prediction = model.predict(img_input)
print("Prediction:", characters[np.argmax(prediction)])