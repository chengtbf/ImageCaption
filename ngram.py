import numpy as np
import pickle

train_file = "/home/chengcheng/ImageCaption/data/train_vector.txt"
valid_file = "/home/chengcheng/ImageCaption/data/valid_vector.txt"
test_file = "/home/chengcheng/ImageCaption/data/test_vector.txt"

train_file = open(train_file, 'r')
valid_file = open(valid_file, 'r')
test_file = open(test_file, 'r')

data = []

def read2data(readfile):
    for line in readfile.readlines():
        strs = line.strip().split()
        if len(strs) == 1:
            continue
        nums = []
        for e in strs:
            nums.append(int(e))
        data.append(nums)

read2data(train_file)
read2data(valid_file)
# read2data(valid_file)
# read2data(valid_file)
read2data(test_file)

# 1 gram

gram_1_dic = {}

def get_prob(x):
    return x / x.sum()

for i in range(len(data)):
    for j in range(len(data[i]) - 1):
        key = data[i][j]
        value_index = data[i][j+1]
        if key not in gram_1_dic:
            gram_1_dic[key] = np.zeros(7000)
        gram_1_dic[key][value_index] += 1


for key in gram_1_dic:
    gram_1_dic[key] = get_prob(gram_1_dic[key])

# np.save('data/1gram.npy', gram_1_dic)
# f = open('data/1gram.pkl', 'wb')
# pickle.dump(gram_1_dic, f, pickle.HIGHEST_PROTOCOL)
