import vocabulary
import inference_wrapper
import configuration
import h5py
import tensorflow as tf
import caption_generator
import math

vocab = vocabulary.Vocabulary("data/dic.txt")
file = h5py.File("/home/chengcheng/dataset/image_caption/feat.hdf5", 'r')
encoded_images = file['valid_set']
valid_list_file = "/home/chengcheng/ImageCaption/Tensorflow/data/valid_list.txt"
check_point_steps = 500000

model = inference_wrapper.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "/home/chengcheng/dataset/image_caption/checkpoints/2/{}.ckpt".format(check_point_steps))

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator.CaptionGenerator(model, vocab)

valid_list_file = open(valid_list_file, 'r')
valid_image_list = []
for line in valid_list_file.readlines():
    valid_image_list.append(line.strip().split()[0])
# output three optional sentences for each image, ranking by probability in decreasing order
with open('/home/chengcheng/dataset/image_caption/inference/2/valid_caption_{}.txt'.format(check_point_steps), 'w') as f:
    for index in range(1000):
        captions = generator.beam_search(sess, encoded_images[index])
        f.write(valid_image_list[index])
        # print("Captions for image {}".format(valid_image_list[index]))
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            f.write("  %d) %s (p=%f)\n" % (i, sentence, math.exp(caption.logprob)))
            # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

# output inferencing result following requirements in the submitting web page
# with open('/home/chengcheng/dataset/image_caption/inference/2/valid_caption_{}.txt'.format(check_point_steps),
#             'w') as f:
#     for index in range(1000):
#         captions = generator.beam_search(sess, encoded_images[index])
#         caption = captions[0]
#         sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
#         full_str = "".join(sentence)
#         f.write(valid_image_list[index] + "\n")
#         for word in full_str:
#             f.write(' {}'.format(word))
#         f.write('\n')
#         if (index + 1) % 100 == 0:
#             print(repr(index + 1))
