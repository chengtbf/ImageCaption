import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math
import json

vocab = vocabulary.Vocabulary("data/dic.txt")
file = h5py.File("/home/chengcheng/dataset/image_caption/feat.hdf5", 'r')
encoded_images = file['valid_set']
valid_list_file = "/home/chengcheng/ImageCaption/data/valid_list.txt"
check_point_steps = 800000
label_image_num = 500
unlabel_image_num = 0

check_point_path = "/home/chengcheng/dataset/image_caption/checkpoints/{}_{}/{}.ckpt"

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               check_point_path.format(label_image_num, unlabel_image_num,check_point_steps))

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)

valid_list_file = open(valid_list_file, 'r')
valid_image_list = []
for line in valid_list_file.readlines():
    valid_image_list.append(line.strip().split()[0])
# output three optional sentences for each image, ranking by probability in decreasing order
# with open('/home/chengcheng/dataset/image_caption/inference/3/valid_caption_{}.txt'.format(check_point_steps), 'w') as f:

result_list = []

for index in range(1000):
    captions = generator.beam_search(sess, encoded_images[index])
    # if encoded_images[index] != valid_image_list[index]:
    #    print(encoded_images[index], valid_image_list[index])

    # print("Captions for image {}".format(valid_image_list[index]))
    for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        result_list.append({"image_id" : index, "caption" : sentence})
        # f.write("  %d) %s (p=%f)\n" % (i, sentence, math.exp(caption.logprob)))
        # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


result_file = open("infer_result/{}_{}_valid_result_{}.json".format(label_image_num, unlabel_image_num,check_point_steps),"w")
json.dump(result_list, result_file)
