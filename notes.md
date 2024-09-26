
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

