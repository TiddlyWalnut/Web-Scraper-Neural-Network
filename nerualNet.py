from tensorflow.keras.applications.densenet import preprocess_input, DenseNet201, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# grab model
base_model = DenseNet201(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
  layer.trainable = False

# Compile and train model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Load data using Keras
batch_size = 16
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  directory="/content/drive/My Drive/data/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size    
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  directory="/content/drive/My Drive/data/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Loading and preprocessing images
def preprocess(img, x):
  return preprocess_input(float(img))
new_train = train_ds.map(preprocess)
new_val = val_ds.map(preprocess)

img = image.load_img("flower.jpeg", target_size=(224,224))
mod = image.img_to_array(img)
mod = np.expand_dims(mod, axis=0)
mod = preprocess_input(mod)

preds = model.predict(mod)
print(class_names)
print(len(preds[0]))
print(preds[0])
print(f"I think:\nChance of {class_names[1]} is: {preds[0][0]}\nChance of {class_names[2]} is: {preds[0][1]}\nChance of {class_names[3]} is: {preds[0][2]}\nChance of {class_names[4]} is: {preds[0][3]}\nChance of {class_names[5]} is: {preds[0][4]}\n")