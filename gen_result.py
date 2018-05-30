import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

conf = configuration.MyConfig()

vocab = vocabulary.Vocabulary("data/dic.txt")
file = h5py.File("data/feat.hdf5", 'r')
encoded_images = file['valid_set']
valid_list_file = "data/valid_list.txt"
train_step = conf.train_step
checkpoint_steps = conf.original_train_steps + (train_step - 1) * conf.interval_train_steps

check_point_path = "train_log/{}.ckpt".format(checkpoint_steps)

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               check_point_path)

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1, use_ngram=conf.use_ngram_gen_result)

valid_list_file = open(valid_list_file, 'r')
valid_image_list = []
for line in valid_list_file.readlines():
    valid_image_list.append(line.strip().split()[0])
# output three optional sentences for each image, ranking by probability in decreasing order
# with open('/home/chengcheng/dataset/image_caption/inference/3/valid_caption_{}.txt'.format(checkpoint_steps), 'w') as f:

result_list = []

for index in range(1000):
    captions = generator.beam_search(sess, encoded_images[index])
    # if encoded_images[index] != valid_image_list[index]:
    #    print(encoded_images[index], valid_image_list[index])
    if index % 100 == 0:
        print(index, '...Done')
    # print("Captions for image {}".format(valid_image_list[index]))
    for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        result_list.append({"image_id" : index, "caption" : sentence})
        # f.write("  %d) %s (p=%f)\n" % (i, sentence, math.exp(caption.logprob)))
        # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


result_file = open("infer_result/{}_{}_valid_result_step{}_checkpoint{}_gram{}_scalar{}.json".format(conf.label_image_size, conf.unlabel_image_size, conf.train_step, checkpoint_steps, conf.n_gram, conf.n_gram_scalar),"w")
# result_file = open("infer_result/1000_0_valid_result_280k_2.json","w")
json.dump(result_list, result_file)
