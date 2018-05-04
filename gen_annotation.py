import json
import os

valid_file = "/home/chengcheng/ImageCaption/data/valid_vector.txt"
valid_list = "/home/chengcheng/ImageCaption/data/valid_list.txt"
dic_file = "/home/chengcheng/ImageCaption/data/dic.txt"

valid_file = open(valid_file, 'r')
valid_list = open(valid_list, 'r')
dic_file = open(dic_file, 'r')

valid_list = list(valid_list.readlines())
valid_list = [line.strip().split()[0] for line in valid_list]
dic = dict([(x, y) for (y, x) in enumerate(valid_list)])

vocab_list = list(dic_file.readlines())
vocab_list = [line.strip().split()[0] for line in vocab_list]


anno = {"annotations":[],"images":[],"type":"captions"}
image_id = 9999999
index = 0

for line in valid_file.readlines():
    strs = line.strip().split()
    if len(strs) == 1:
        image_id = dic[strs[0]]
    else:
        sentence = ""
        for i in range(1, len(strs) - 1):
            word = vocab_list[int(strs[i])]
            sentence = sentence + " " + word
        one_anno = {'id': index, 'caption':sentence, 'image_id': image_id}
        anno["annotations"].append(one_anno)
        image = {'id': image_id }
        anno["images"].append(image)
        index+=1


valid_anno = open("/home/chengcheng/ImageCaption/data/valid_anno.json","w")
json.dump(anno, valid_anno)

