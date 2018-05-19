from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

label_image_num = 500
unlabel_image_num = 5500
train_step = 1
check_point_steps = 100000 * (1 + train_step)
# set up file names and pathes
resultFile="infer_result/{}_{}_valid_result_feat2_it{}_{}.json"
annFile='data/valid_anno.json'
subtypes=['result', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
[resultFile.format(label_image_num, unlabel_image_num,train_step, check_point_steps) for subtype in subtypes]

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
# cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

eval_result = open('data/eval_result.txt','a')
eval_result.write("{}label_{}unlabel_{}step_feat2_it{}_result:\n".format(label_image_num, unlabel_image_num, train_step,check_point_steps))
print('final result:')

for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
    eval_result.write(str(metric)+ ": " + str(score) + " |")
eval_result.write("\n");
eval_result.close();



