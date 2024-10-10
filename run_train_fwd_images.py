

# defaults:
#   - vision_model: resnet_autoencoder  # Options: vae, resnet_autoencoder
#   - fwd_model: mlp_fwd_model
#   
# nz: 12
# nu: 2
# nzu: 14
# data_path: "./new_data_all_2024-09-05.pt_downsampled.pt"
# # data_path: "./new_data_all_2024-09-05.pt"
# image_size: 64
#
# training:
#   lr: 1e-4
#   train_num_steps: 100000
#   fix_encoder: false
#   noise_z: 1e-2
#   noise_u: 1e-2
#   noise_img: 1e-2
#   weight_decay: 0.0
#   z_predict: 1e-1
#   z_reg: 1e-5
#   batch_size: 16
#   num_workers: 1
#   seed: 42
#   img_recon: 1.0
#   img_predict: 1.0
#   test_size_ratio: 0.05  # 5% for testing

# small script to run a grid search over hyperparameters.
