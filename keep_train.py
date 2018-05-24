import configuration
import os
import h5py
import tensorflow as tf
import show_and_tell_model
import read_data

train_step = 1
single_train_step_checkpoints = 300000
checkpoint_steps = single_train_step_checkpoints * train_step
label_image_num = 1000
unlabel_image_num = 5000

checkpoint_path = "train_log/{}.ckpt".format(checkpoint_steps)

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

iter = read_data.DataIterator(encoded_image_path="data/feat.hdf5",
                              caption_vector_path="train_log/{}_infer_train_vector.txt".format(checkpoint_steps),image_size=label_image_num + unlabel_image_num)

# gpu_config = tf.ConfigProto(device_count = {'GPU': 1})
# sess = tf.InteractiveSession(config=gpu_config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sess = tf.InteractiveSession()

model = show_and_tell_model.ShowAndTellModel(model_config, mode="train")
model.build()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=5)

# if tf.gfile.IsDirectory(checkpoint_path):
#     checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
#     if not checkpoint_path:
#         raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

saver.restore(sess, checkpoint_path)

loss_stored = []

# step = epoch * train_data_set / batch_size

# with tf.device('/gpu:1'):
for i in range(train_step * single_train_step_checkpoints, (train_step + 1) * single_train_step_checkpoints):
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
'''
        if(i+1)% 100000 ==0:
            iter.infer_unlabel_image_captions('train_log/{}.ckpt'.format(i+1), vocab)
'''

with open('train_log/loss.txt', 'a') as f:
    for e in loss_stored:
        f.write(repr(e))
        f.write('\n')
