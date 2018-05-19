import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math
import json

vocab = vocabulary.Vocabulary("data/dic.txt")
file = h5py.File("data/feat.hdf5", 'r')
encoded_images = file['train_set']
train_list_file = "data/train_list.txt"
train_vector_file = "data/train_vector.txt"
train_step = 2
checkpoint_steps = 100000 * train_step

label_image_num = 500
unlabel_image_num = 5500

checkpoint_path = "train_log/{}.ckpt".format(checkpoint_steps)

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               checkpoint_path)

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)

train_list_file = open(train_list_file, 'r')
train_image_list = []
for line in train_list_file.readlines():
    train_image_list.append(line.strip().split()[0])
# output three optional sentences for each image, ranking by probability in decreasing order
# with open('/home/chengcheng/dataset/image_caption/inference/3/train_caption_{}.txt'.format(check_point_steps), 'w') as f:

caption_vector_path="train_log/{}_infer_train_vector.txt".format(checkpoint_steps)

label_file = open(caption_vector_path, 'w')

index = -1
for line in open(train_vector_file):
    strs = line.strip().split()
    if len(strs) == 1:
        index = index + 1
        image_name = strs[0]
        if train_image_list[index] != image_name:
            print('wrong! ' + train_image_list[index] + " != " + image_name)
        if index == label_image_num:
            break
    label_file.write(line)

for index in range(index, index + unlabel_image_num):
    captions = generator.beam_search(sess, encoded_images[index])
    # if encoded_images[index] != train_image_list[index]:
    #    print(encoded_images[index], train_image_list[index])
    label_file.write(train_image_list[index]+"\n")

    # print("Captions for image {}".format(train_image_list[index]))
    for i, caption in enumerate(captions):
        sentence = [str(w) for w in caption.sentence]
        sentence = " ".join(sentence)
        label_file.write(sentence + "\n")
        # result_list.append({"image_id" : index, "caption" : sentence})
        # Ignore begin and end words.
        # sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        # sentence = " ".join(sentence)
        # result_list.append({"image_id" : index, "caption" : sentence})
        # f.write("  %d) %s (p=%f)\n" % (i, sentence, math.exp(caption.logprob)))
        # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
label_file.close()
