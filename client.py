# https://cyberaiguy.com/building-attacking-mnist

import numpy as np
import matplotlib.pyplot as plt
import requests
from keras.datasets import mnist
from PIL import Image
import io
import imageio

#optionally, get an image out from the dataset

GetNewImageFromMNIST = True
if GetNewImageFromMNIST == True:
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Combine the train and test sets if you want to select from the entire dataset
    all_images = np.concatenate((train_images, test_images), axis=0)
    # Generate a random index
    random_index = np.random.choice(all_images.shape[0])
    # Select the image
    random_image = all_images[random_index]
    # Display the image
    plt.imshow(random_image, cmap='gray')
    plt.title(f"Random MNIST digit: {random_index}")
    plt.axis('off')  # Hide the axis to focus on the image
    plt.show()

    # Save the image to the filesystem
    filename = f"mnist_digit_{random_index}.png"
    imageio.imwrite(filename, random_image)
    print(f"Image saved as {filename}")

else:
    filename = 'mnist_digit_67876.png'

# The path to the image you want to send
image_path = filename
server_url = 'http://localhost:42000/predict'


# Open the image in binary mode
with open(image_path, 'rb') as image_file:
    # The POST request with the binary data of the image
    image_binary = image_file.read()

#send the OG image
response = requests.post(server_url, data=image_binary)
print(response.text)

#setup an attack routine
def add_random_noise(imageIn, noise_level=0.1):
    # Assuming image is a numpy array of shape (height, width, channels)
    # Add random noise to the image
    perturbation = noise_level * np.random.randn(*imageIn.shape)
    perturbed_image = imageIn + perturbation
    # Clip the image pixel values to be between 0 and 1
    perturbed_image = np.clip(perturbed_image, 0.0, 1.0)
    return perturbed_image


image = Image.open(io.BytesIO(image_binary))
image_array = np.array(image)

perturbed_image_array = add_random_noise(image_array,.0005)
perturbed_image = Image.fromarray(perturbed_image_array.astype('uint8'), 'L')

perturbed_image_path='perturbed_image.png'
perturbed_image.save(perturbed_image_path)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title(f"Original")
plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
plt.subplot(1, 2, 2)
plt.title(f"Modified")
plt.imshow(perturbed_image, cmap='gray')  # Use cmap='gray' for grayscale images
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

with open(perturbed_image_path, 'rb') as image_file:
    perturbed_image_binary = image_file.read()

#send the perturbed image
response = requests.post(server_url, data=perturbed_image_binary)
print(response.text)
