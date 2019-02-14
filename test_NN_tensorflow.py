# coding: utf-8
import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
#######################################################################
# Load Data
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "D:\Data_Minig\seance9_neuralNetwork\\"
train_data_directory = os.path.join(ROOT_PATH, "traffic\Training\\")
test_data_directory = os.path.join(ROOT_PATH, "traffic\Testing\\")

images, labels = load_data(train_data_directory)

#convert to array
images = np.array(images)
labels = np.array(labels)
##################################################################################
"""
# Print the `images` dimensions
print(images.ndim)
# Print the number of `images`'s elements
print(images.size)
# Print the first instance of `images`
images[0]

# Print the `labels` dimensions
print(labels.ndim)
# Print the number of `labels`'s elements
print(labels.size)
# Count the number of labels
print(len(set(labels)))"""
##################################################################################

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
#plt.show()
##################################################################################
# Rescale the images in the `images` array since they are not the same size
images_28 = [transform.resize(image, (28, 28)) for image in images]

# Convert `images28` to an array
images_28 = np.array(images_28)

# Convert `images28` to grayscale
images_28 = rgb2gray(images_28)
####################################################################################
# Determine the (random) indexes of the images that you want to see
randoms =[]
for i in range(0,4):
    randoms.append(random.randint(1, 4575))

print(randoms)
traffic_signs = randoms
# Fill out the subplots with the random images that you defined
def plot_images(images):
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]],cmap='gray')
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                      images[traffic_signs[i]].min(),
                                                      images[traffic_signs[i]].max()))
    plt.show()
######################################################################################
#plot_images(images)
#plot_images(images_28)
######################################################################################
# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data 28x28= 784
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#########################################
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)
##########################################
#train
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images_28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')
#############################################
#Evaluating Model
# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))
##################################################
#Test
# Pick 10 random images
sample_indexes = random.sample(range(len(images_28)), 10)
sample_images = [images_28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()
sess.close()
"""
# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run())

# Close the session
sess.close()"""