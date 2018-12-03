import tensorflow as tf
import model
# import utils
import cv2
import numpy as np
from numpy import matrix


network_type = "CC"

train_x1_lcc = cv2.imread("small_dataset_train/_00008_LEFT_CC.png",0)
train_x1_rcc = cv2.imread("small_dataset_train/_00008_RIGHT_CC.png",0)
train_y = matrix( [[0,0,1]] )
if network_type == "CC":
    train_x1_lcc = cv2.resize(train_x1_lcc, (2000, 2600))
    train_x1_rcc = cv2.resize(train_x1_rcc, (2000, 2600))
    train_x1_lcc = np.array(train_x1_lcc).reshape(1, 2000, 2600,1)
    train_x1_rcc = np.array(train_x1_rcc).reshape(1, 2000, 2600,1)
else:
    train_x1_lmlo = cv2.resize(train_x1_lcc, (2000, 2600))
    train_x1_rmlo = cv2.resize(train_x1_rcc, (2000, 2600))
    train_x1_lmlo = np.array(train_x1_lmlo).reshape(1, 2000, 2600,1)
    train_x1_rmlo = np.array(train_x1_rmlo).reshape(1, 2000, 2600,1)
#print(resized_image.shape, type(resized_image))
training_iters = 200
learning_rate = 0.001
batch_size = 1

# Network parameters
#n_input = "none"
n_classes = 3

x1_cc= tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])
x2_cc= tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])

x1_mlo= tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])
x2_mlo= tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])

if network_type == "CC":
    x = (x1_cc,x2_cc)
else:
    x = (x1_mlo,x2_mlo)

#x = tf.placeholder(dtype=tf.int32, shape = None, name=None)

y = tf.placeholder(tf.float32, shape=(1,3))

prediction = model.baseline(x, network_type)

print(prediction)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate model node
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)

# config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter("apg_git_repo/training_graph/output", sess.graph)
    if network_type == "CC":
        for i in range(training_iters):
            opt = sess.run(optimizer, feed_dict={x1_cc:train_x1_lcc,x2_cc:train_x1_rcc,y:train_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x1_cc:train_x1_lcc,x2_cc:train_x1_rcc,y:train_y})
            print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("Optimization Finished!")

        summary_writer.close()

    else:
        for i in range(training_iters):
            opt = sess.run(optimizer, feed_dict={x1_mlo: train_x1_lmlo, x2_mlo: train_x1_rmlo, y: train_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x1_mlo: train_x1_lmlo, x2_mlo: train_x1_rmlo, y: train_y})
            print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("Optimization Finished!")

        summary_writer.close()
    #savePath = saver.save(sess, 'E:/training_graph/agp_birads.ckpt')
