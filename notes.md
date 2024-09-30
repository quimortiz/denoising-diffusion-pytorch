
0vsivm is cnn "/home/quim/code/resnet-18-autoencoder/data/65nzbo/model_65nzbo.ckpt" 
gets loss 0.0610


sjc0bq is good. lets load the model

# Sep 18
is good:
results/i2n6ce/interp-95000.png
python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --size 32 --img_interp_weight 1e-1



python train_only_encoder_v2.
py --z_diff 1e-4 --z_interp_weight 1e-2  --size 64 --img_interp_weight 1e-1 --weight_deca
y 1e-7
results/74idb0/interp-95000.png



quim@mango ~/c/denoising-diffusion-pytorch (main) [SIGINT]> python train_only_encoder_v2
.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e-4  --size 64 --img_interp_weight
 1e-1 --weight_decay 1e-7
saving to results/g31iq2/interp-30000.png


# Sep 19



## New encoder models that look good.

saving to results/6ghoqo/interp-95000.png

python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e-4  --size 64 --img_interp_weight 1e-1 --weight_decay 1e-7

##

python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e-4  --size 64  --weight_decay 1e-8 --resnet

saving to results/5knvwh/interp-95000.png




not bad:
results/0y2s5y-fine/sample-cond-1000.png                                                                                                                                                                                    


## 

This is a generative model that looks good.

```
python  example_1d_seq_encoder_wu.py --size 64  --cond --z_diff 1.    --lr 1e-4 --mod_
lr --recon_weight 1e-12 --weight_decay 1e-9  --pretrained --fix_encoder | tee /tmp/(date "+log_file_%Y-%m-%d_%H-%M-%S.txt")
```

normal conditioning seems better than increasing the dimensionality


results/sv3j5d-fine/sample-cond-100.png


# Sep 20.

###
this encoder with resnet is also good.
results/la90ra/interp-330000.png

python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e-4  --size 64  --weight_decay 1e-8 --resnet --train_num_steps 1000000

maybe try with less regularization on z? and remove decay?


###
also not bad,
python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e-4  --size 64  --weight_decay 1e-8  --train_num_steps 1000000

saving to results/zn9sms/interp-995000.png


# Sep 23

###

I had this from before
results/5knvwh/model-95000


this seems fine!
6mmx65 
-- 690000

What about this? 
la90ra
-- 655000
-- 490000

Idea: 
Train only on z's.
afterwards, train also in img loss for some steps.
even the decoder?
but now that i have a good decoder, could i even try to just use input space?


# Sep 27

this model is using the new autoencoder stuff.
results/ue71o0/interp-995000.png

# Sep 30

I have to have a free 4 layer. not clear aobut what is better, only NLP or only Linear.

(deep) quim@mango ~/c/denoising-diffusion-pytorch (main)> python train_only_encoder_v2.py --z_diff 1e-4 --z_interp_weight 1e-2  --z_weight 1e
-4  --size 64   --train_num_steps 1000000
Namespace(exp_id='39uqbf', pretrained=False, lr=0.0001, recon_weight=0.0, z_weight=0.0001, z_diff=0.0001, cond=False, size=64, mod_lr=False, 
cond_combined=False, y_cond_as_x=False, weight_decay=0.0, train_num_steps=1000000, resnet=False, train_u=False, z_interp_weight=0.01, img_int
erp_weight=0.01)
/home/quim/envs/deep/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated sin
ce 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/quim/envs/deep/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None`
 for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_We
ights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Model size: 0.0890 GB
len dataset 10664
10664
 layer 4  fixed False

TODO: 
Lets try to train the original model with the pretrained stuff!
