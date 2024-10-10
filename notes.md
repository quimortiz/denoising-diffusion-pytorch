
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


# Oct 1

This is also good, encoder from pretrained Resnet.

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
device cuda




TODO: evaluate this with the multiple step stuff!
this generative model is very good, but not very multimodal.

saving to results/ibt1fa/sample-imgs-cond-00999.png
sampling loop time step: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 73.20it/s]
saving to results/ibt1fa/sample-imgs-00999.png
step: 1000000 / 1000000, loss: 0.0013
Current Learning Rate: 0.000100
sampling loop time step: 100%|████████████████████████████████████████████████████████████
saving to results/ibt1fa/sample-imgs-cond-01000.png
sampling loop time step: 100%|████████████████████████████████████████████████████████████
███████████████| 100/100 [00:01<00:00, 73.97it/s]
saving to results/ibt1fa/sample-imgs-01000.png
sampling loop time step:   0%|                                                            


what about this one?

this one also doesnt seem multimodal, but is very precise
saving to results/8qljoo/sample-imgs-cond-00969.png
step: 969000 / 1000000, loss: 0.0018
Current Learning Rate: 0.000100


this has more multimodal, but few steps.
python   example_1d_seq.py --pretrained --resnet --size 64 --train_u --noise_y 5e-2 |  tee log/(date "+log_file_%Y-
%m-%d_%H-%M-%S.txt")
saving to results/5lxgoy/sample-imgs-00044.png


encoders:

python train_only_encoder_v2.py --z_diff 1e-3   --z_weight 1e-4  --size 64   --tra
in_num_steps 1000000 --resnet --img_interp_weight 0. --z_interp_weight 0.

this doesn't go below 0.003


this is going better
(deep) quim@mango ~/c/denoising-diffusion-pytorch (main) [SIGINT]> python train_only_encoder_v2.py --z_diff 1e-3   --z_weight 1e-5  --size 64
   --train_num_steps 1000000 --resnet_vae_new --img_interp_weight 0. --z_interp_weight 0.



# Oct 8


this is good
(deep) quim@mango ~/c/denoising-diffusion-pytorch (main)> python train_only_encoder_v2.py --z_diff 1e-4   --z_weight 1e-4  --size 64   --train_num_steps 1000000 --resnet --img_interp_weight 0. --z_interp_weight 0. --noise_z .05

Namespace(exp_id='z91yo7', pretrained=False, lr=0.0001, recon_weight=0.0, z_weight=0.0001, z_diff=0.0001, cond=False, size=64, mod_lr=False, 
cond_combined=False, y_cond_as_x=False, weight_decay=0.0, train_num_steps=1000000, resnet=True, resnet_vae_new=False, train_u=False, z_interp
_weight=0.0, img_interp_weight=0.0, noise_z=0.05, noise_img=0.01)


Average z norm 1.6695843935012817
Raw loss: img_loss 0.00013649938046000898 z_loss 0.3524981439113617 z_traj_loss 0.3319145143032074 z_interp_loss 0.01341493334621191 img_int
erp_loss 0.0
Weighted Loss: img_loss 0.00013649938046000898 z_loss 3.5249814391136173e-05 z_traj_loss 3.319145143032074e-05 z_interp_loss 0.0 img_interp_
loss 0.0
step 998000 loss 0.00017160331481136382
Average z norm 1.6522274017333984
Raw loss: img_loss 0.00010690793715184554 z_loss 0.3452676832675934 z_traj_loss 0.30168604850769043 z_interp_loss 0.012444701045751572 img_i
nterp_loss 0.0
Weighted Loss: img_loss 0.00010690793715184554 z_loss 3.452676832675934e-05 z_traj_loss 3.0168604850769046e-05 z_interp_loss 0.0 img_interp_
loss 0.0

issue: maybe the z is bigger than one in some components?



this is also goods:
(deep) quim@mango ~/c/denoising-diffusion-pytorch (main) [SIGINT]> python train_only_encoder_v2.py --z_diff 1e-4   --z_weight 5e-4  --size 6
4   --train_num_steps 1000000 --resnet --img_interp_weight 0. --z_interp_weight 0. --noise_z .1

Average z norm 1.575462818145752
Raw loss: img_loss 0.00034911310649476945 z_loss 0.3164539039134979 z_traj_loss 0.4669402837753296 z_interp_loss 0.021165788173675537 img_in
terp_loss 0.0
Weighted Loss: img_loss 0.00034911310649476945 z_loss 0.00015822695195674897 z_traj_loss 4.669402837753296e-05 z_interp_loss 0.0 img_interp_
loss 0.0



diffusion models

(deep) quim@mango ~/c/denoising-diffusion-pytorch (main) [SIGINT]> python   example_1d_seq.py --pretrained --resnet --size 64  --train_u  | 
 tee log/(date "+log_file_%Y-%m-%d_%H-%M-%S.txt")

Caution, setting us to zero for now

us Minimum per channel: tensor([0., 0.])
us Maximum per channel: tensor([0., 0.])
loading model from  results/la90ra/model-490000.pt ...

step: 1000000 / 1000000, loss: 0.0009
Current Learning Rate: 0.000100
sampling loop time step: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 74.22it/s]
saving to results/2e231l/sample-imgs-cond-01000.png
sampling loop time step: 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 91.82it/s]
saving to results/2e231l/sample-imgs-01000.png
sampling loop time step:   0%|                                                                                     | 0/100 [00:00<?, ?it/s]
good but still not very multimodal.



About learning a model

python -m pdb  train_fwd.py --pretrained --train_num_steps 1000000 --lr 1e-3 --resnet --size 64

this works well, example:
results/5oweul/sample-imgs-02994.png


# Oct 9

This seems good:
python train_only_encoder_v2.py --z_diff 5e-4   --z_weight 1e-4  --size 64   --train_num_steps 1000000 --resnet --img_interp_weight 0. --z_interp_weight 0. --noise_z .05  --z_dim 12
saving to results/y6owhr/interp-460000.png
t
there was bug and z dim is still 8.
last step is step 572000 

