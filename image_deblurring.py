import cv2
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape

import os
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, target_size=(64, 64)):
    processed_images = []
    for img in images:
        # Resize image to target size
        img = cv2.resize(img, target_size)
        # Normalize pixel values to the range [0, 1]
        img = img / 255.0
        processed_images.append(img)
    return np.array(processed_images)

blurry_folder_path = 'D:/archive/motion_blurred'
clear_folder_path = 'D:/archive/sharp'

blurry_images = load_images_from_folder(blurry_folder_path)
clear_images = load_images_from_folder(clear_folder_path)

target_size = (64, 64)  
processed_blurry_images = preprocess_images(blurry_images, target_size)
processed_clear_images = preprocess_images(clear_images, target_size)

print("Processed Blurry Images Shape:", processed_blurry_images.shape)
print("Processed Clear Images Shape:", processed_clear_images.shape)

from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model

def build_generator(input_shape):
    noise_input = Input(shape=input_shape)
    x = Dense(128 * 16 * 16)(noise_input)
    x = Reshape((16, 16, 128))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    generated_image = Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')(x)

    return Model(noise_input, generated_image)


input_shape = (100,)  
generator = build_generator(input_shape)
generator.summary()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64, 64, 3), padding='same'))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

input_shape = (64, 64, 3)
discriminator = build_discriminator()
discriminator.build(input_shape)
discriminator.summary()

def data_generator(blurry_images, clear_images, batch_size):
    while True:
        indices = np.random.choice(len(blurry_images), batch_size, replace=False)
        batch_blurry = blurry_images[indices]
        batch_clear = clear_images[indices]
        yield batch_blurry, batch_clear
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)

gan.compile(optimizer='adam', loss='binary_crossentropy')


epochs = 1000
batch_size = 32

# Create data generator
data_gen = data_generator(processed_blurry_images, processed_clear_images, batch_size)

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

for epoch in range(epochs):
    blurry_batch, clear_batch = next(data_gen)
    fake_images = generator.predict(np.random.rand(batch_size, 100))
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(clear_batch, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.rand(batch_size, 100)
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")

    if epoch % save_interval == 0:
        save_generated_images(epoch, generator)

import os
import matplotlib.pyplot as plt

def save_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 64, 64, 3)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')

    plt.savefig(f'generated_images/generated_image_epoch_{epoch}.png')

#final 
new_blurry_image = cv2.imread('C:/Users/shakt/OneDrive/Pictures/Screenshots/Tokyo.png')
new_blurry_image = cv2.cvtColor(new_blurry_image, cv2.COLOR_BGR2RGB)
new_blurry_image = new_blurry_image / 255.0

enhanced_image = generator.predict(np.expand_dims(new_blurry_image, axis=0))[0]

plt.subplot(1, 2, 1)
plt.imshow(new_blurry_image)
plt.title('Blurry Image')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image)
plt.title('Enhanced Image')

plt.show()