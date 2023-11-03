# https://cyberaiguy.com/building-attacking-mnist

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import io


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def train_model(model_path):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def preprocess(images, labels):
        # Convert the images to float32
        images = tf.cast(images, tf.float32)
        # Normalize the images to [0, 1]
        images = images / 255.0
        # Add a channel dimension, images will have shape (28, 28, 1)
        images = tf.expand_dims(images, -1)
        return images, labels

    # Apply the preprocess function to our training and testing data
    ds_test = ds_test.map(preprocess)
    ds_train = ds_train.map(preprocess)

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_test = ds_test.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ## create and tune the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    #save the model 
    tf.keras.models.save_model(model, model_path)

    # Assuming `ds_test` is your test dataset, and it's batched
    # Take 10 examples from the test set
    for images, labels in ds_test.take(1):
        # Select 10 images and labels
        test_images = images[:10]
        test_labels = labels[:10]
        predictions = model.predict(test_images)

    # Display the images and the model's predictions
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i].numpy().squeeze(), cmap=plt.cm.binary)
        plt.xlabel(f"Actual: {test_labels[i].numpy()}")
        plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.tight_layout()
    plt.show()

model_path = 'mnist-saved-model'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"The model file {model_path} does not exist. Training now. ")
    # train the model if it doesn't exist yet 
    train_model(model_path)

model = load_model(model_path)



class RequestHandler(BaseHTTPRequestHandler):
    model = load_model('mnist-saved-model')

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            print("[-] Recieved request.. ")
                
            try:
                # Use PIL to open the image and convert it to the expected format
                image = Image.open(io.BytesIO(post_data)).convert('L')
                image = image.resize((28, 28))
                image = np.array(image) / 255.0
                image = image.reshape(1, 28, 28, 1)
                print("[-] Making prediction from submitted image.. ")
                # Make prediction
                prediction = self.model.predict(image)
                predicted_class = np.argmax(prediction, axis=1)
                print(f'This image most likely is a {predicted_class[0]} with a probability of {np.max(prediction)}.')


                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                resp = f'This image most likely is a ' + str(predicted_class[0])  + ' with a probability of {:.3%}'.format(np.max(prediction))
                self.wfile.write(json.dumps(resp).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def runServer(server_class=HTTPServer, handler_class=RequestHandler, port=42000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving HTTP on port {port}...')
    httpd.serve_forever()

runServer()
