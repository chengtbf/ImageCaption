from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import configuration

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

conf = configuration.MyConfig()
# check_point_steps = 100000 * (1 + train_step)
train_step = conf.train_step
checkpoint_steps = conf.original_train_steps + (train_step - 1) * conf.interval_train_steps
# set up file names and pathes
resultFile = "infer_result/{}_{}_valid_result_step{}_checkpoint{}_gram{}_scalar{}.json".format(conf.label_image_size, conf.unlabel_image_size, conf.train_step, checkpoint_steps, conf.n_gram, conf.n_gram_scalar)
# resultFile = "infer_result/{}_{}_valid_result_step{}_{}.json".format(conf.label_image_size, conf.unlabel_image_size, conf.train_step, checkpoint_steps)
# resultFile="infer_result/{}_{}_valid_result_feat2_it{}_{}.json".format(label_image_num, unlabel_image_num,train_step, check_point_steps)
# resultFile = "infer_result/1000_0_valid_result_280k_2.json"
annFile='data/valid_anno.json'
'''
subtypes=['result', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
[resultFile.format(label_image_num, unlabel_image_num,train_step, check_point_steps) for subtype in subtypes]
print(resFile)
'''
# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resultFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
# cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

eval_result = open('data/eval_result.txt','a')
eval_result.write("{}label_{}unlabel_{}step_it{}_gram{}_scalar{}_result:\n".format(conf.label_image_size, conf.unlabel_image_size, train_step, checkpoint_steps, conf.n_gram, conf.n_gram_scalar))
print('final result:')

for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
    eval_result.write(str(metric)+ ": " + str(score) + " |")
eval_result.write("\n");
eval_result.close();
