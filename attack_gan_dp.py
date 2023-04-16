import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

img_height = 64
img_width = 64
epsilon = 10.0
mode = 'dp'

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fairRR(arr, eps, num_int, num_bit, mode = 'dp'):
    r = arr.shape[1]
    num_pt = arr.shape[0]
    
    def float_to_binary(x, m=num_int, n=num_bit - num_int - 1):
        x_abs = np.abs(x)
        x_scaled = round(x_abs * 2 ** n)
        res = '{:0{}b}'.format(x_scaled, m + n)
        if x >= 0:
            res = '0' + res
        else:
            res = '1' + res
        return res

    # binary to float
    def binary_to_float(bstr, m=num_int, n=num_bit - num_int - 1):
        sign = bstr[0]
        bs = bstr[1:]
        res = int(bs, 2) / 2 ** n
        if int(sign) == 1:
            res = -1 * res
        return res

    def string_to_int(a):
        bit_str = "".join(x for x in a)
        return np.array(list(bit_str)).astype(int)

    def join_string(a, num_bit=num_bit, num_feat=r):
        res = np.empty(num_feat, dtype="S10")
        # res = []
        for i in range(num_feat):
            # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
            res[i] = "".join(str(x) for x in a[i * num_bit:(i + 1) * num_bit])
        return res
    
    def alpha_tr1(r, eps, l):
        return np.exp( ( eps - r*eps*(l-1) ) /(2*r*l) )

    def alpha(r, eps, l):
        nu = 2*( np.sqrt( 6*np.log(10) /(2*r) ) )
        sum_ = 0
        for k in range(l):
            sum_ += np.exp(2 * eps*k / l)
        return np.sqrt(((1-nu)*eps + r*l) / (2*r * sum_))

    alpha_ = alpha_tr1(r=r, eps=eps, l=num_bit) if mode == 'dp' else alpha(r=r, eps=eps, l=num_bit)
    
    float_to_binary_vec = np.vectorize(float_to_binary)
    binary_to_float_vec = np.vectorize(binary_to_float)

    feat_tmp = float_to_binary_vec(arr)
    feat = np.apply_along_axis(string_to_int, 1, feat_tmp)
    
    index_matrix = np.array(range(num_bit))
    index_matrix = np.tile(index_matrix, (num_pt, r))
    p = 1 / (1 + alpha_ * np.exp(index_matrix * eps / num_bit))
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    print(feat[0][:10])
    print(perturb[0][:10])
    perturb_feat = (perturb + feat) % 2
    print(perturb_feat[0][:10])
    perturb_feat = np.apply_along_axis(join_string, 1, perturb_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    print(arr[0][:2])
    print(perturb_feat[0][:2])
    return perturb_feat

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

temp_plain = np.reshape(true_label, newshape=(true_label.shape[0], true_label.shape[1]*true_label.shape[2]*true_label.shape[3]))
perturb_feat = fairRR(arr=temp_plain, eps=epsilon, num_int=5, num_bit=10, mode = mode)
true_label = tf.convert_to_tensor(np.reshape(perturb_feat, newshape=true_label.shape))

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
plt.savefig(f'eps_{epsilon}_mode_{mode}.jpg')
