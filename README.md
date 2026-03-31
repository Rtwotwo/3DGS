# :rocket: Project Guide :rocket:

The 3DRepo repository is mainly used to reproduce currently cutting-edge 3D reconstruction technologies, including implicit technologies such as Nerf, MipNerf, Tri-MipNerf, and explicit technical models such as 3D Gaussian Splatting, 2D Gaussian Splatting, and Mip-Splatting. The datasets used in the project mainly include datasets like Nerf-Synthetic and Mip-360 (if you need to download the datasets, you can click [Nerf Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and [Mip-360](https://jonbarron.info/mipnerf360/) to download them). Additionally, the project also includes processing methods and visualization operations for these data. You can view the performance of these models on real datasets by watching the effects of specific synthetic perspectives or checking specific metrics such as PSNR, SSIM, and LPIPS.

## :wrench: 1.Experiment Results :wrench:

We choose the Nerf-Synthetic and Mip-360 dataset to train the 3D Gaussian Splatting, 2D Gassian Splatting, Gaussian Opacity Fields, Mip-Splatting, Analytic-Splatting and 3DGRUT algorithms and make Single-Train-Single-Test(STST), Single-Train-Multiple-Test(STMT) experiments to get metrics about PSNR, SSIM, LPIPS to compare the performance of algorithms. If you need to view all the test results, you can find and check them in detail at [results.xlsx](assets/results.xlsx).The detailed results of these experiments are as follows:

We first present the performance of different algorithms on the NeRF-Synthetic dataset. The algorithms tested include 3DGRT, 3DGUT, GOF, 2DGS, 3DGS, Analytic-Splatting, Mip-Splatting, 3D-Mip-Splatting and other methods, which were evaluated at different resolutions including R1/R2/R4/R8 under the STST training and testing framework. Three types of metric data were obtained: PSNR, SSIM and LPIPS. Below, we only show the test results at resolution R1 under the STST paradigm on the NeRF-Synthetic dataset.

Conclusions: Based on a comprehensive analysis of all metrics, 3DGS, as the baseline method, exhibits the weakest performance in reconstruction quality. In contrast, 3D-Mip-Splatting achieves the best overall performance with an average PSNR of 34.18, SSIM of 0.9706, and the lowest LPIPS of 0.0287. Its core advantage lies in effectively balancing high-frequency details and rendering stability through an anti-aliasing mechanism. In comparison, Analytic-Splatting performs well in some static scenes but has limitations on complex materials. GOF and the 3DG series methods show robust performance without significant breakthroughs, while 2DGS lags in overall accuracy due to limited representation capability. This indicates that the current trend in technical development has shifted from simply improving reconstruction accuracy toward addressing frequency mismatch and aliasing issues in multi-scale rendering.

| PSNR | chair | drums | ficus | hotdog | lego | materials | mic | ship | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3d-mip-splatting | 36.2017 | 26.3664 | 36.5691 | 38.3151 | 36.932 | 30.6889 | 36.427 | 31.9751 | 34.1844 |
| Mip-Splatting | 35.6777 | 26.3531 | 35.8899 | 38.2419 | 36.3788 | 30.6634 | 37.0175 | 31.7289 | 33.9939 |
| Analytic-Splatting | 36.5045 | 26.3452 | 36.3378 | 38.0487 | 36.4348 | 27.7219 | 31.6219 | 31.2186 | 33.0292 |
| GOF | 35.7882 | 26.2738 | 35.8567 | 37.2613 | 35.9214 | 30.3162 | 36.6210 | 31.6022 | 33.7051 |
| 3DGUT | 35.4330 | 25.9220 | 36.3880 | 38.0630 | 36.2950 | 30.3750 | 36.4430 | 31.6190 | 33.8173 |
| 3DGRT | 35.3840 | 25.7220 | 36.5060 | 37.8640 | 36.7150 | 30.3870 | 35.8320 | 31.6760 | 33.7608 |
| 2DGS | 34.7748 | 24.5630 | 35.7941 | 36.9893 | 32.7877 | 30.1241 | 34.2083 | 30.0538 | 32.4119 |
| 3DGS | 31.9301 | 24.9148 | 29.0489 | 36.5010 | 32.3826 | 29.6901 | 34.6608 | 29.5509 | 31.0849 |

| SSIM | chair | drums | ficus | hotdog | lego | materials | mic | ship | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3d-mip-splatting | 0.9887 | 0.9555 | 0.9887 | 0.9860 | 0.9845 | 0.9615 | 0.9922 | 0.9074 | 0.9706 |
| Mip-Splatting | 0.9881 | 0.9558 | 0.9879 | 0.9860 | 0.9840 | 0.9615 | 0.9930 | 0.9064 | 0.9703 |
| Analytic-Splatting | 0.9884 | 0.9553 | 0.9893 | 0.9858 | 0.9838 | 0.9490 | 0.9835 | 0.9072 | 0.9678 |
| 3DGRT | 0.9870 | 0.9520 | 0.9890 | 0.9860 | 0.9850 | 0.9610 | 0.9910 | 0.9080 | 0.9699 |
| GOF | 0.9880 | 0.9558 | 0.9877 | 0.9854 | 0.9827 | 0.9593 | 0.9924 | 0.9077 | 0.9699 |
| 3DGUT | 0.9880 | 0.9530 | 0.9880 | 0.9860 | 0.9830 | 0.9600 | 0.9920 | 0.9060 | 0.9695 |
| 2DGS | 0.9856 | 0.9340 | 0.9872 | 0.9835 | 0.9657 | 0.9576 | 0.9868 | 0.8799 | 0.9600 |
| 3DGS | 0.9829 | 0.9410 | 0.9528 | 0.9839 | 0.9756 | 0.9502 | 0.9869 | 0.8952 | 0.9586 |

| LPIPS | chair | drums | ficus | hotdog | lego | materials | mic | ship | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3d-mip-splatting | 0.0096 | 0.0360 | 0.0103 | 0.0182 | 0.0137 | 0.0356 | 0.0064 | 0.0995 | 0.0287 |
| Mip-Splatting | 0.0110 | 0.0365 | 0.0112 | 0.0185 | 0.0149 | 0.0357 | 0.0061 | 0.1026 | 0.0296 |
| Analytic-Splatting | 0.0106 | 0.0366 | 0.0099 | 0.0194 | 0.0144 | 0.0486 | 0.0132 | 0.1046 | 0.0321 |
| GOF | 0.0113 | 0.0371 | 0.0114 | 0.0210 | 0.0168 | 0.0376 | 0.0066 | 0.1050 | 0.0308 |
| 3DGRT | 0.0150 | 0.0500 | 0.0130 | 0.0240 | 0.0170 | 0.0460 | 0.0100 | 0.1230 | 0.0373 |
| 3DGUT | 0.0130 | 0.0500 | 0.0130 | 0.0270 | 0.0200 | 0.0460 | 0.0090 | 0.1260 | 0.0380 |
| 2DGS | 0.0130 | 0.0607 | 0.0124 | 0.0250 | 0.0310 | 0.0411 | 0.0118 | 0.1324 | 0.0409 |
| 3DGS | 0.0235 | 0.0592 | 0.0433 | 0.0313 | 0.0317 | 0.0640 | 0.0264 | 0.1297 | 0.0511 |

Subsequently, we conducted the same tests on the Mip-360 dataset. Considering the varying computational resource requirements of different algorithms, we did not test the 3D-Mip-Splatting algorithm at the original scale. The corresponding test results are presented below. In the evaluation on the STST-Images2-r1 dataset, Mip-Splatting demonstrated significant comprehensive advantages, taking the overall lead with an average PSNR of 27.36, SSIM of 0.8067, and the lowest LPIPS of 0.2299. It performed particularly well in texture-rich scenes such as bonsai and kitchen, reflecting its robustness in handling complex lighting and high-frequency details.

Although 3d-mip-splatting had a slightly lower PSNR (27.22) than Mip-Splatting, the gap was minimal, and its SSIM (0.8039) and LPIPS (0.2352) were also at a top-tier level, indicating that its 3D smoothing mechanism effectively maintains structural consistency. In comparison, 3DGS, as the baseline method, delivered a moderate performance (PSNR 27.05). It retained some detail advantages in scenes such as garden and kitchen through the original Gaussian representation but was still inferior to the improved algorithms overall. GOF achieved a PSNR (26.84) close to that of 3DGS but was slightly weaker in SSIM (0.7922) and LPIPS (0.2560), with a noticeable drop in reconstruction quality especially in distant scenes like treehill. Analytic-Splatting and 2DGS each exposed their limitations: the former suffered from low PSNR in scenes such as flowers and treehill (20.50, 22.14), while the latter obtained SSIM values of only 0.6495 and 0.7246 on dynamic structures such as bicycle and stump, revealing insufficient capabilities in complex geometric modeling.Overall, Mip-Splatting and its variants achieved the optimal balance between accuracy and stability in real-world scene reconstruction through anti-aliasing and multi-scale optimization. The original 3DGS remains competitive in specific scenarios but requires further improvements to address challenges in complex environments.

| PSNR | bicycle | bonsai | counter | flowers | garden | kitchen | room | stump | treehill | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mip-Splatting | 24.7536 | 32.5686 | 29.0561 | 21.2278 | 26.6776 | 31.6578 | 31.6769 | 26.7407 | 21.9132 | 27.3636 |
| 3d-mip-splatting | 24.3869 | 31.8519 | 29.2464 | 21.1940 | 26.2964 | 31.6399 | 31.6378 | 26.8371 | 21.8924 | 27.2203 |
| 3DGS | 24.2836 | 32.0962 | 29.0327 | 20.4780 | 26.4492 | 31.2103 | 31.5862 | 26.1699 | 22.1713 | 27.0530 |
| Analytic-Splatting | 24.2149 | 32.3960 | 29.1004 | 20.4993 | 24.6710 | 31.3336 | 31.5927 | 26.2148 | 22.1399 | 26.9070 |
| GOF | 24.5728 | 31.5742 | 28.6946 | 21.1308 | 25.2310 | 30.7656 | 30.8709 | 26.7741 | 21.9882 | 26.8447 |
| 2DGS | 23.6028 | 31.2095 | 28.0469 | 19.8371 | 25.7906 | 29.9978 | 30.8394 | 25.3845 | 21.8618 | 26.2856 |

| SSIM | bicycle | bonsai | counter | flowers | garden | kitchen | room | stump | treehill | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mip-Splatting | 0.7408 | 0.9460 | 0.9130 | 0.5841 | 0.8255 | 0.9307 | 0.9250 | 0.7751 | 0.6201 | 0.8067 |
| 3d-mip-splatting | 0.7357 | 0.9407 | 0.9117 | 0.5819 | 0.8202 | 0.9280 | 0.9244 | 0.7734 | 0.6187 | 0.8039 |
| GOF | 0.7286 | 0.9371 | 0.9022 | 0.5723 | 0.7694 | 0.9166 | 0.9160 | 0.7695 | 0.6183 | 0.7922 |
| 3DGS | 0.6924 | 0.9399 | 0.9066 | 0.5323 | 0.8084 | 0.9253 | 0.9185 | 0.7472 | 0.6146 | 0.7872 |
| Analytic-Splatting | 0.6895 | 0.9409 | 0.9074 | 0.5327 | 0.7425 | 0.9255 | 0.9185 | 0.7478 | 0.6149 | 0.7800 |
| 2DGS | 0.6495 | 0.9284 | 0.8901 | 0.4986 | 0.7836 | 0.9133 | 0.9055 | 0.7246 | 0.5927 | 0.7651 |

| LPIPS | bicycle | bonsai | counter | flowers | garden | kitchen | room | stump | treehill | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mip-Splatting | 0.2432 | 0.1881 | 0.1868 | 0.3480 | 0.1563 | 0.1190 | 0.2021 | 0.2713 | 0.3539 | 0.2299 |
| 3d-mip-splatting | 0.2499 | 0.1965 | 0.1906 | 0.3540 | 0.1610 | 0.1220 | 0.2067 | 0.2757 | 0.3602 | 0.2352 |
| GOF | 0.2764 | 0.1982 | 0.2031 | 0.3669 | 0.2373 | 0.1368 | 0.2165 | 0.2927 | 0.3761 | 0.2560 |
| Analytic-Splatting | 0.3332 | 0.2080 | 0.2020 | 0.4307 | 0.2838 | 0.1280 | 0.2220 | 0.3275 | 0.4132 | 0.2832 |
| 3DGS | 0.3284 | 0.2060 | 0.2012 | 0.4285 | 0.1847 | 0.1268 | 0.2197 | 0.3270 | 0.4114 | 0.2704 |
| 2DGS | 0.3977 | 0.2299 | 0.2335 | 0.4769 | 0.2360 | 0.1485 | 0.2449 | 0.3810 | 0.4619 | 0.3123 |

## :mag: 2.Colmap Tool :mag:

Colmap first reconstructs the sparse 3D structure and camera poses through SfM, then uses MVS to densify the sparse points, and finally outputs a dense point cloud or mesh. Overall, Colmap's entire pipeline is accelerated by a CPU/GPU hybrid, with high precision, robustness, and open-source availability, making it the "offline reconstruction baseline" in academia and industry. The specific process for installing the GPU version of Colmap on Linux (taking Ubuntu 22.04 as an example) is as follows:

```bash
# Install the compilation dependencies
sudo apt update
sudo apt install -y gcc-11 g++-11
git clone https://github.com/colmap/colmap.git
cd colmap

#If you encounter compilation issues with PoseLib, you need to manually download the PoseLib source code package.
cd ~/colmap
wget https://github.com/PoseLib/PoseLib/archive/f119951fca625133112acde48daffa5f20eba451.zip
# Unzip to the diectional floder
unzip -q f119951fca625133112acde48daffa5f20eba451.zip
mv PoseLib-f119951fca625133112acde48daffa5f20eba451 \
   build/_deps/poselib-src

# Rebuild and trigger CMake
cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11

# Continuing to compilation
ninja -j4
sudo ninja install
colmap patch_match_stereo --help | grep gpu
```


