import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math


class Convolutional():

    def __init__(self, data = None):
        self.data = data

        self.filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
        self.num_filters1 = 16         # There are 16 of these filters.

        self.filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
        self.num_filters2 = 36         # There are 36 of these filters.

        # Fully-connected layer.
        self.fc_size = 128             # Number of neurons in fully-connected layer.

        self.img_size = 20
        self.img_size_flat = self.img_size * self.img_size

        self.img_shape = (self.img_size, self.img_size)

        self.num_channels = 4
        self.num_classes = self.img_size/2 * (self.img_size/2 - 1) / 2 + self.img_size/2

        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')

        self.x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, self.num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        self.train_batch_size = 64

        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=self.x_image,
                           num_input_channels=self.num_channels,
                           filter_size=self.filter_size1,
                           num_filters=self.num_filters1,
                           use_pooling=True)

        layer_conv2, weights_conv2 = \
            self.new_conv_layer(input=layer_conv1,
                           num_input_channels=self.num_filters1,
                           filter_size=self.filter_size2,
                           num_filters=self.num_filters2,
                           use_pooling=True)

        layer_flat, num_features = self.flatten_layer(layer_conv2)

        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=self.fc_size,
                                 use_relu=True)

        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                 num_inputs=self.fc_size,
                                 num_outputs=self.num_classes,
                                 use_relu=False)

        y_pred = tf.nn.softmax(layer_fc2)

        y_pred_cls = tf.argmax(y_pred, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=self.y_true)

        cost = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())

        self.total_iterations = 0

    def set_data(self, data):
        self.data = data

    def define_layers(self):

        self.test_batch_size = 256

        # Counter for total number of iterations performed so far.
        self.total_iterations = 0

        self.print_test_accuracy()

        self.optimize(num_iterations=1)

        self.print_test_accuracy()

        self.optimize(num_iterations=99)

        self.print_test_accuracy()



    def plot_images(self, images, cls_true, cls_pred=None):
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        #equivalent to y intercept
        #constant value carried over across matrix math
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(self, input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    def flatten_layer(layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self, input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def optimize(self, num_iterations):

        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(self.total_iterations,
                       self.total_iterations + num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = self.data.train.next_batch(self.train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if i % 100 == 0:
                # Calculate the accuracy on the training-set.
                acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

                # Print it.
                print(msg.format(i + 1, acc))

        # Update the total number of iterations performed.
        self.total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def plot_example_errors(self, cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = self.data.test.images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = self.data.test.cls[incorrect]

        # Plot the first 9 images.
        self.plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])

    def print_test_accuracy(self, show_example_errors=False):

        # Number of images in the test-set.
        num_test = len(self.data.test.images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + self.test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = self.data.test.images[i:j, :]

            # Get the associated labels.
            labels = self.data.test.labels[i:j, :]

            # Create a feed-dict with these images and labels.
            feed_dict = {self.x: images,
                         self.y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = self.session.run(self.y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Convenience variable for the true class-numbers of the test-set.
        cls_true = self.data.test.cls

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            self.plot_example_errors(cls_pred=cls_pred, correct=correct)


class SimpleConvolutional:
    def __init__(self):

        self.img_size = 20

        self.num_classes = (self.img_size / 2 * (self.img_size / 2 - 1) / 2 + self.img_size / 2) * 10

        self.batch_size = 64

        self.img_size_flat = self.img_size * self.img_size

        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')

        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')

        self.keep_rate = 0.8

        self.keep_prob = tf.placeholder(tf.float32)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def maxpool2d(x):
        #                        size of window         movement of window
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_neural_network(self, x):
        weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                   'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.nn.relu(tf.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = tf.maxpool2d(conv1)

        conv2 = tf.nn.relu(tf.conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = tf.maxpool2d(conv2)

        fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    def train_neural_network(self, x):
        prediction = self.convolutional_neural_network(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 10
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(self.data.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = self.mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({self.x: self.data.test.images, self.y: self.data.test.labels}))

