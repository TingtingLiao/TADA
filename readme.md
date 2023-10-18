<p align="center">
<h2 align="center">TADA! Text to Animatable Digital Avatars</h2>
<p align="center">
<a href="https://github.com/TingtingLiao"><strong>Tingting Liao*</strong></a>
·  
<a href="https://xyyhw.top/"><strong>Hongwei Yi*</strong></a>
·
<a href="http://xiuyuliang.cn/"><strong>Yuliang Xiu</strong></a>
·
<a href="https://me.kiui.moe/"><strong>Jiaxiang Tang</strong></a>
·
<a href="https://github.com/huangyangyi/"><strong>Yangyi Huang</strong></a>
·
<a href="https://justusthies.github.io/"><strong>Justus Thies</strong></a>
·
<a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
<br>
    * Equal Contribution
</p>

<h3 align="center">3DV 2024</h4>
 
<p align="center">
    <a href='https://arxiv.org/abs/2308.10899'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://tada.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/TADA-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=orange' alt='Project Page'>
    <a href="https://youtu.be/w5rdcPQWktE"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/w5rdcPQWktE?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
    </a>
    <p/>
<br/>
<div align="center">
    <img src="https://tada.is.tue.mpg.de/media/upload/teaser.png" alt="Logo" width="100%">
</div>


**NEWS (2023.9.24)**:

* Using Omnidata normal prediction model to improve the normal&image consistency.


https://github.com/TingtingLiao/TADA-code/assets/45743512/248d70ab-f755-46f1-bb4f-1a8468f30901


https://github.com/TingtingLiao/TADA-code/assets/45743512/d7ad2b0f-6c29-46ba-9090-d91d027a5a6b




# TODO
- [x] Adding Omnidata normal supervision for texture and geometry consistency
- [ ] Supporting single image reconstruction 
- [ ] Adding shading code 

[//]: # (TADA takes text as input and produce holistic animatable 3D avatars with high-quality geometry and texture. )
[//]: # (It enables creation of large-scale digital character assets that are ready for animation and rendering, while also being easily editable through natural language. )
 
# Install
- System requirement: Unbuntu 20.04 
- Tested GPUs: RTX4090, A100, V100 
- Compiler: gcc-7.5 / g++-7.5 
- Python 3.9, CUDA 11.5  

```bash[readme.md](..%2F..%2F%E4%B8%8B%E8%BD%BD%2Freadme.md)
git clone git@github.com:TingtingLiao/TADA.git
cd TADA

conda env create --file environment.yml
conda activate tada 
pip install -r requirements.txt
 
cd smplx
python setup.py install 
```
- Download [TADA Extra Data](https://download.is.tue.mpg.de/download.php?domain=tada&resume=1&sfile=tada_extra_data.zip) 
- Download [SMPL-X Model](http://smpl-x.is.tue.mpg.de/) and put it in the directory ./data/smplx
- Download [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for depth and normal prediction.
```bash
mkdir data/omnidata 
cd data/omnidata 
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
```

 
 
# Usage 

### Training 

The results will be save in $workspace. Please change it in the config/*.yaml files.
```python 
# single prompt training    
python -m apps.run --config configs/configs/tada_w_dpt.yaml.yaml --text "Aladdin in Aladdin" 

# multiple prompts training
bash scripts/run.sh data/prompt/fictional.txt 1 10 configs/tada.yaml
``` 

### Animation 
- Download [AIST](https://aistdancedb.ongaaccel.jp/) or generate motions from [TalkShow](https://github.com/yhw-yhw/TalkSHOW) and [MotionDiffusion](https://github.com/GuyTevet/motion-diffusion-model). 
```  
python -m apps.anime --subject "Abraham Lincoln" --res_dir your_result_path
``` 


# Tips and Tricks
* using an appropriate learning rate for SMPL-X shape is important to learn accurate shape. 

# Acknolowget 
This work is build upon Stable DreamFusion, many thanks to the author [Jiaxiang Tang](https://github.com/ashawkey) and many other contributors.
* [Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion)  
``` 
@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}
```  
* [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/)
```
@article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
    
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10975--10985},
  year = {2019}
}
```
* [TalkShow](https://talkshow.is.tue.mpg.de/), [AIST]() and [MotionDiffusion](). 
``` 
@inproceedings{yi2023generating,
title={Generating Holistic 3D Human Motion from Speech},
  author={Yi, Hongwei and Liang, Hualin and Liu, Yifei and Cao, Qiong and Wen, Yandong 
and Bolkart, Timo and Tao, Dacheng and Black, Michael J},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
pages={469-480},
month={June}, 
year={2023} 
}
```

# Citation

```bibtex
@article{liao2023tada,
title={TADA! Text to Animatable Digital Avatars},
author={Liao, Tingting and Yi, Hongwei and Xiu, Yuliang and Jiaxiang Tang and Huang, Yangyi and Thies, Justus and Black, Michael J},
journal={ArXiv},
month={Aug}, 
year={2023} 
}
```

