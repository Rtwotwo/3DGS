A# :rocket: Project Guide :rocket:

The 3DRepo repository is mainly used to reproduce currently cutting-edge 3D reconstruction technologies, including implicit technologies such as Nerf, MipNerf, Tri-MipNerf, and explicit technical models such as 3D Gaussian Splatting, 2D Gaussian Splatting, and Mip-Splatting. The datasets used in the project mainly include datasets like Nerf-Synthetic and Mip-360 (if you need to download the datasets, you can click [Nerf Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and [Mip-360](https://jonbarron.info/mipnerf360/) to download them). Additionally, the project also includes processing methods and visualization operations for these data. You can view the performance of these models on real datasets by watching the effects of specific synthetic perspectives or checking specific metrics such as PSNR, SSIM, and LPIPS.

## :wrench: 1.Experiment Results :wrench:

We choose the Nerf-Synthetic and Mip-360 dataset to train the 3D Gaussian Splatting, 2D Gassian Splatting, Gaussian Opacity Fields, Mip-Splatting, Analytic-Splatting and 3DGRUT algorithms and make Single-Train-Single-Test(STST), Single-Train-Multiple-Test(STMT) experiments to get metrics about PSNR, SSIM, LPIPS to compare the performance of algorithms. The detailed results of these experiments are as follows:
___

