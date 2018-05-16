import h5py
import numpy as np
import inference_wrapper
import configuration
import tensorflow as tf
import caption_generator

class DataIterator:
    def __init__(self, encoded_image_path, caption_vector_path, image_size):
        # structure of h5 file ['test_set', 'train_set', 'validation_set']
        file = h5py.File(encoded_image_path, 'r')
        encoded_images = file['train_set']
        print("train_size :" + str(len(encoded_images)))


        train_list_file = open('data/train_list.txt', 'r')
        train_image_list = []
        for line in train_list_file.readlines():
            train_image_list.append(line.strip().split()[0])

        images_and_captions = []
        index = -1
        image_name = ""
        for line in open(caption_vector_path):
            strs = line.strip().split()
            if len(strs) == 1:
                index = index + 1
                image_name = strs[0]
                if train_image_list[index] != image_name:
                    print('wrong! ' + train_image_list[index] + " != " + image_name)
                if index == image_size:
                    break
            else:
                nums = []
                for e in strs:
                    nums.append(int(e))
                images_and_captions.append([encoded_images[index], nums])

        self.images_and_captions = images_and_captions

        self.iter_order = np.random.permutation(len(images_and_captions))
        self.cur_iter_index = 0
        print("Finish loading data")

        print('Training set size is {}'.format(len(images_and_captions)))

    def next_batch(self, batch_size):
        images = []
        captions = []

        for i in range(batch_size):
            # print(self.cur_iter_index+i)
            image, caption = self.images_and_captions[self.iter_order[self.cur_iter_index]]
            images.append(image)
            captions.append(caption)
            self.cur_iter_index += 1
            if self.cur_iter_index >= len(self.images_and_captions):
                self.iter_order = np.random.permutation(len(self.images_and_captions))
                self.cur_iter_index = 0

        input_seqs, target, masks = self.build_caption_batch(captions)

        return images, input_seqs, target, masks

    def build_caption_batch(self, captions):
        input_seqs = []
        target = []
        masks = []
        max_len = 0
        for caption in captions:
            if len(caption) > max_len:
                max_len = len(caption)

        max_len = max_len - 1

        for caption in captions:
            want_len = len(caption) - 1
            input_seqs.append(caption[0:want_len] + [0] * (max_len - want_len))
            target.append(caption[1:(want_len + 1)] + [0] * (max_len - want_len))
            masks.append([1] * want_len + [0] * (max_len - want_len))

        return input_seqs, target, masks
    # transform raw captions to three parts
    # an input caption meets following requirements:
    #   1) start with 1 (the start flag)
    #   2) end with 2 (the end flag)
    # assume the batch size is 4, and the captions are:
    # [1, 3, 4, 5, 6, 2]
    # [1, 5, 6, 7, 2]
    # [1, 5, 3, 2]
    # [1, 7, 9, 10, 2]
    # then the outputs are
    # input_seqs:
    # [1, 3, 4, 5, 6]
    # [1, 5, 6, 7, 0]
    # [1, 5, 3, 0, 0]
    # [1, 7, 9, 10, 0]
    # target_seqs:
    # [3, 4, 5, 6, 2]
    # [5, 6, 7, 2, 0]
    # [5, 3, 2, 0, 0]
    # [7, 9, 10, 2, 0]
    # input_masks:
    # [1, 1, 1, 1, 1]
    # [1, 1, 1, 1, 0]
    # [1, 1, 1, 0, 0]
    # [1. 1, 1, 1, 0]



'''
    def infer_unlabel_image_captions(self, check_point_path, vocab):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                       check_point_path)
            sess = tf.InteractiveSession()
            restore_fn(sess)

            generator = caption_generator.CaptionGenerator(model, vocab, beam_size=3)

            self.unlabel_images_and_captions = []
            for i in range(self.unlabel_size):
                captions = generator.beam_search(sess, self.unlabel_images[i])
                for caption in captions:
                    self.unlabel_images_and_captions.append([self.unlabel_images[i], caption])

            print('unlabel images captions size is {}'.format(len(self.unlabel_images_and_captions)))
            self.total_images_and_captions  = self.label_images_and_captions + self.unlabel_images_and_captions
            self.iter_order = np.random.permutation(len(self.total_images_and_captions))
            self.cur_iter_index = 0
'''


