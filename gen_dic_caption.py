token_file = "/home/chengcheng/dataset/image_caption/label/Flickr8k.lemma.token.txt"
train_images = "/home/chengcheng/dataset/image_caption/label/Flickr_8k.trainImages.txt"
valid_images = "/home/chengcheng/dataset/image_caption/label/Flickr_8k.devImages.txt"
test_images = "/home/chengcheng/dataset/image_caption/label/Flickr_8k.testImages.txt"
dic_file = "/home/chengcheng/ImageCaption/data/dic.txt"
train_file = "/home/chengcheng/ImageCaption/data/train_vector.txt"
valid_file = "/home/chengcheng/ImageCaption/data/valid_vector.txt"
test_file = "/home/chengcheng/ImageCaption/data/test_vector.txt"
train_list_file = "/home/chengcheng/ImageCaption/data/train_list.txt"
valid_list_file = "/home/chengcheng/ImageCaption/data/valid_list.txt"
test_list_file = "/home/chengcheng/ImageCaption/data/test_list.txt"


'''
dic = {}
dic["<S>"] = 0
dic["</S>"] = 1
dic["<UNK>"] = 2
'''
dic = ["<UNK>","<S>","</S>"]
word2id = {}
word2id["<S>"] = 1
word2id["</S>"] = 2
word2id["<UNK>"] = 0

token_file = open(token_file)
dic_file = open(dic_file, 'w')
train_images = open(train_images)
valid_images = open(valid_images)
test_images = open(test_images)
train_file = open(train_file, 'w')
valid_file = open(valid_file, 'w')
test_file = open(test_file, 'w')
train_list_file = open(train_list_file, 'w')
valid_list_file = open(valid_list_file, 'w')
test_list_file = open(test_list_file, 'w')

train_image_list = []
test_image_list = []
valid_image_list = []

for line in train_images.readlines():
    train_image_list.append(line.strip().split()[0])


for line in valid_images.readlines():
    valid_image_list.append(line.strip().split()[0])


for line in test_images.readlines():
    test_image_list.append(line.strip().split()[0])

# print(train_image_list)


word_size = 3
lines = token_file.readlines()
for line in lines:
    strs = line.strip().split()
    for i in range(1,len(strs)):
        word = strs[i].lower()
        if word not in word2id:
            dic.append(word)
            word2id[word] = word_size
            word_size += 1
'''
        if not dic.has_key(word):
            dic[word] = i
            i += 1

for k,v in dic.items():
    dic_file.write(k + " " + repr(v) + "\n")
'''
print (word_size)

for i in range(len(dic)):
    dic_file.write(dic[i] + " " + repr(i) + "\n")

dic_file.close()

image_now = ""
file_now = train_file
list_file_now = train_list_file

real_train_list = []
real_valid_list = []
real_test_list = []

import os

image_list = os.listdir("/home/chengcheng/dataset/image_caption/image/Flicker8k_Dataset/")

for line in lines:
    strs = line.strip().split()
    image = strs[0].split('#')[0]
    if image not in image_list:
        print image
        continue
    if image != image_now:
        image_now = image
        if image_now in train_image_list:
            file_now = train_file
            list_file_now = train_list_file
        elif image_now in valid_image_list:
            file_now = valid_file
            list_file_now = valid_list_file
        elif image_now in test_image_list:
            file_now = test_file
            list_file_now = test_list_file
        else :
            print("can't find :" + image)
            list_file_now = train_list_file
            file_now = train_file
        file_now.write(image + "\n")
        list_file_now.write(image + "\n")

    file_now.write("1 ")
    for i in range(1,len(strs)):
        word = strs[i].lower()
        id = word2id[word]
        file_now.write(repr(id) + " ")
    file_now.write("2\n")

train_file.close()
valid_file.close()
test_file.close()
train_list_file.close()
valid_list_file.close()
test_list_file.close()
