import tensorflow as tf
import test_model as model
#import utils
import cv2



train_x1_lcc = cv2.imread("C:/Users/archit.kushwaha.AGILiAD\Documents\Training_data\L-CC/_00008_LEFT_CC.png",0)
train_x1_rcc = cv2.imread("C:/Users/archit.kushwaha.AGILiAD\Documents\Training_data\R-CC/_00008_RIGHT_CC.png",0)
train_y = 2
resized_image = cv2.resize(train_x1_lcc, (28, 28))
print(resized_image.shape, type(resized_image))
training_iters = 200
learning_rate = 0.001
batch_size = 1

# Network parameters
#n_input = "none"
n_classes = 3

x = tf.placeholder(dtype=tf.int32, shape = None, name=None)

y = tf.placeholder(dtype=tf.float32, shape=None, name=None)

prediction = model.baseline(resized_image)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter("E:/training_graph/output", sess.graph)
    for i in range(training_iters):
        opt = sess.run(optimizer, feed_dict={x:resized_image,y:train_y})

    summary_writer.close()
