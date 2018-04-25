token_file = "/home/chengcheng/image_caption/label/Flickr8k.lemma.token.txt"
dic_file = "/home/chengcheng/image_caption/label/Flickr8k.dic.txt"

'''
dic = {}
dic["<S>"] = 0
dic["</S>"] = 1
dic["<UNK>"] = 2
'''
dic = ["<S>","</S>","<UNK>"]

token_file = open(token_file)
dic_file = open(dic_file, 'w')

i = 3
lines = token_file.readlines()
for line in lines:
    strs = line.strip().split()
    for i in range(1,len(strs)):
        word = strs[i].lower()
        if word not in dic:
            dic.append(word)
'''
        if not dic.has_key(word):
            dic[word] = i
            i += 1

for k,v in dic.items():
    dic_file.write(k + " " + repr(v) + "\n")
'''

for i in range(len(dic)):
    dic_file.write(dic[i] + " " + repr(i) + "\n")

dic_file.close()

