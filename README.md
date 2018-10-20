
# SRDenseNet-Tensorflow
Tensorflow implemetation of SRDensenet 
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_SRDenseNet_tensorflow/master/asset/srdensenet_model.png" width="600">
</p>


## Prerequisites
 * python 3.x
 * Tensorflow > 1.5
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * matplotlib
 * argparse

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3
 * This code supports only RGB color images (demo type) and Ychannel of YCbCr (eval type) 
 * This code supports data augmentation (rotation and mirror flip)
 * This code supports custom dataset


## Usage
```
usage: main.py [-h] [--exp_tag EXP_TAG] [--gpu GPU] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--patch_size PATCH_SIZE]
               [--base_lr BASE_LR] [--lr_min LR_MIN]
               [--num_denseblock NUM_DENSEBLOCK]
               [--num_denselayer NUM_DENSELAYER] [--growth_rate GROWTH_RATE]
               [--lr_decay_rate LR_DECAY_RATE] [--lr_step_size LR_STEP_SIZE]
               [--scale SCALE] [--checkpoint_dir CHECKPOINT_DIR]
               [--cpkt_itr CPKT_ITR] [--save_period SAVE_PERIOD]
               [--train_subdir TRAIN_SUBDIR] [--test_subdir TEST_SUBDIR]
               [--infer_subdir INFER_SUBDIR] [--infer_imgpath INFER_IMGPATH]
               [--type {eval,demo}] [--c_dim C_DIM]
               [--mode {train,test,inference,test_plot}]
               [--result_dir RESULT_DIR] [--save_extension {jpg,png}]

Namespace(base_lr=0.0001, batch_size=32, c_dim=3, checkpoint_dir='checkpoint', cpkt_itr=0, epoch=80, exp_tag='SRDenseNet tensorflow. Implemented by Dohyun Kim', gpu=1, growth_rate=16, infer_imgpath='monarch.bmp', infer_subdir='Custom', lr_decay_rate=0.1, lr_min=1e-06, lr_step_size=30, mode='train', num_denseblock=8, num_denselayer=8, patch_size=33, result_dir='result', save_extension='.jpg', save_period=1, scale=3, test_subdir='Set5', train_subdir='291', type='demo')
```

 * For training, `python3 main.py --mode train --type demo --check_itr 0` [set 0 for training from scratch, -1 for latest]
 * For testing, `python 3main.py --mode test --type demo`
 * For inference with cumstom dataset, `python3 main.py --mode inference --infer_imgpath 3.bmp` [result will be generated in ./result/inference]
 * For running tensorboard, `tensorboard --logdir=./board` then access localhost:6006 with your browser

## Result

</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_SRDenseNet_tensorflow/master/asset/srdensenet_tb.png" width="600">
</p>



## References
* [SRDenseNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) : reference paper



## ToDo
* support eval mode
* support tensorboard history
* link pretrained models
* link dataset

## Author
Dohyun Kim



