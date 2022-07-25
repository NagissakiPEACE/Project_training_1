Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@NagissakiPEACE 
NagissakiPEACE
/
Project_training_1
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Project_training_1/Classification
@NagissakiPEACE
NagissakiPEACE Create Classification
Latest commit e41aa06 1 minute ago
 History
 1 contributor
127 lines (105 sloc)  4.69 KB

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from google.colab import files
%matplotlib inline 

from google.colab import drive
drive.mount('/content/drive')

#batch_size=15
image_size=(100, 100) #розмір зображень у вибірці

train_dataset = image_dataset_from_directory('drive/MyDrive/Data_Set/Training', #сворення тестового набору
                                             subset='training',
                                             seed=5,
                                             validation_split=0.2,
                                             image_size=image_size,
                                             label_mode= 'binary') 

validation_dataset = image_dataset_from_directory('drive/MyDrive/Data_Set/Training', #сворення валідаційного набору
                                             subset='validation',
                                             seed=42,
                                             validation_split=0.2,
                                             image_size=image_size,
                                             label_mode= 'binary')

train_dataset.class_names #відображення класів


#візуалізація даних
plt.figure(figsize=(8, 8))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
  
    plt.axis("off")

#використання бібліотеки tensorflow.keras для оптимального використання ЦП
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#Створення послідовної моделі
model = Sequential()
# Згортковий шар
model.add(Conv2D(15, (10, 10), padding='same', 
                 input_shape=(100, 100, 3), activation='relu'))
#шар підвибірки
model.add(MaxPooling2D(pool_size=(4, 4)))
# Згортковий шар
model.add(Conv2D(60, (10, 10), activation='relu', padding='same'))
#шар підвибірки
model.add(MaxPooling2D(pool_size=(4, 4)))
# Згортковий шар
model.add(Conv2D(120, (10, 10), activation='relu', padding='same'))
#шар підвибірки
model.add(MaxPooling2D(pool_size=(2, 4)))
# повнозвязана частина нейронної мережі для класифікації
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
#вихідний шар з кількістю по класах
model.add(Dense(2, activation='softmax'))

#налаштування втрат та метрик
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

#тренування моделі
history = model.fit(train_dataset, 
                    validation_data=validation_dataset,
                    epochs=20)

 scores = model.evaluate(train_dataset, verbose=1)

plt.figure(figsize=(12, 8))

plt.plot(history.history['accuracy'], 
         label='Кількість правильних відповідей на навчальному наборі даних')
plt.plot(history.history['val_accuracy'], 
         label='Кількість правильних відповідей на валідаційному наборі даних')
plt.xlabel("Епохи")
plt.ylabel('Правильні відповіді')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], 
         label='Втарти  на навчальному наборі')
plt.plot(history.history['val_loss'], 
         label='Втарти  на валідаційному наборі')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

model.save("clasification.h5")

img_path = "drive/MyDrive/Data_Set/Test/68.jpg"

img = image.load_img(img_path, target_size=(100, 100))
plt.imshow(img)
plt.show()

img = (np.expand_dims (img, 0))
predictions_single = model.predict(img)
print(predictions_single)
