import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

img_height = 64
img_width = 64

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, img

file_dir = '../../Datasets/CelebA/img_align_celeba/000001.jpg'
img, label = process_path(file_dir)
print(f'The shape of the image {img.shape}')

encoder = tf.keras.applications.resnet50.ResNet50(
            input_shape = (224,224,3), 
            weights = 'imagenet', 
            include_top = False, 
            pooling = 'avg'
        )

encoder = tf.keras.applications.resnet50.ResNet50(
            input_shape = (img_height,img_width,3), 
            weights = 'imagenet', 
            include_top = False, 
            pooling = 'avg'
        ).layers[:7]

encoder = tf.keras.Sequential(encoder)
true_label = encoder(tf.expand_dims(img, axis=0), training=False)
recon_img = tf.Variable(tf.expand_dims(tf.random.normal(img.shape, 0, 1, tf.float32), axis=0), trainable=True)
print('Details:', img.shape, true_label.shape, recon_img.shape)

loss_object = tf.keras.losses.MeanSquaredError()
def loss(model, label, img_, training):
    y_ = model(img_, training=training)
    return loss_object(y_true=label, y_pred=y_)

l = loss(encoder, true_label, img_=recon_img, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, targets, inputs, training=False)
    return loss_value, tape.gradient(loss_value, inputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_value, grads = grad(encoder, recon_img, true_label)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip([grads], [recon_img]))
print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(), loss(encoder, true_label, recon_img, training=False).numpy()))

train_loss_results = []

num_epochs = 1000000

for epoch in range(num_epochs):
    loss_value, grads = grad(encoder, recon_img, true_label)
    optimizer.apply_gradients(zip([grads], [recon_img]))
    if epoch % 100000 == 0:
        print(f'epoch {epoch}, loss {loss_value}')

plt.imshow(np.squeeze(recon_img).astype("uint8"))
plt.savefig('clean.jpg')
