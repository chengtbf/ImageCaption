from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import configuration
import show_and_tell_model
import read_data
import os

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

iter = read_data.DataIterator(encoded_image_path="data/feat.hdf5",
                              caption_vector_path="data/train_vector.txt", image_size=500)

# gpu_config = tf.ConfigProto(device_count = {'GPU': 1})
# sess = tf.InteractiveSession(config=gpu_config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sess = tf.InteractiveSession()

model = show_and_tell_model.ShowAndTellModel(model_config, mode="train")
model.build()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=3)

loss_stored = []

# step = epoch * train_data_set / batch_size

# with tf.device('/gpu:1'):
for i in range(100000):
    images, in_seqs, tar_seqs, masks = iter.next_batch(model_config.batch_size)
    loss = model.run_batch(sess, images, in_seqs, tar_seqs, masks)
    #every 100 steps print loss value
    if (i+1) % 100 == 0:
        print('step: {}, loss: {}'.format(i+1, loss))
        loss_stored.append(loss)

    #every 1000 steps save check-point file
    if (i+1) % 20000 == 0:
        print('save... step: {}, loss: {}'.format(i+1, loss))
        save_path = saver.save(sess, 'train_log/{}.ckpt'.format(i+1))

with open('train_log/loss.txt', 'w') as f:
    for e in loss_stored:
        f.write(repr(e))
        f.write('\n')
