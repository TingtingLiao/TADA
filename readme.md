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
 
 

TADA takes text as input and produce holistic animatable 3D avatars with high-quality geometry and texture. 
It enables creation of large-scale digital character assets that are ready for animation and rendering, while also being easily editable through natural language. 

**NEWS (2023.9.24)**:

* Using Omnidata normal prediction model to improve the normal&image consistency.


https://github.com/TingtingLiao/TADA-code/assets/45743512/248d70ab-f755-46f1-bb4f-1a8468f30901


https://github.com/TingtingLiao/TADA-code/assets/45743512/d7ad2b0f-6c29-46ba-9090-d91d027a5a6b


# Install
- System requirement: Unbuntu 20.04 
- Tested GPUs: RTX4090, A100, V100 
- Compiler: gcc-7.5 / g++-7.5 
- Python=3.9, CUDA=11.5, Pytorch=1.12.1

```bash
git clone git@github.com:TingtingLiao/TADA.git
cd TADA

conda env create --file environment.yml
conda activate tada 
pip install -r requirements.txt
 
cd smplx
python setup.py install 

# download omnidata normal and depth prediction model 
mkdir data/omnidata 
cd data/omnidata 
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
```

# Data

- [SMPL-X Model](http://smpl-x.is.tue.mpg.de/) (Need to register, download the SMPLX_NEUTRAL_2020.npz and put it into ./data/smplx/)
- [TADA Extra Data](https://download.is.tue.mpg.de/download.php?domain=tada&resume=1&sfile=tada_extra_data.zip) (Required) Unzip it as directory ./data 
- [TADA 100 Characters](https://drive.google.com/file/d/1rbkIpRmvPaVD9AJeCxWqBBYHkRIwrNmC/view?usp=sharing) (Optional)
- Optional Motion Data  
  - [AIST](https://aistdancedb.ongaaccel.jp/), [AIST++](https://google.github.io/aichoreographer/)
  - [TalkShow](https://github.com/yhw-yhw/TalkSHOW)
  - [MotionDiffusion](https://github.com/GuyTevet/motion-diffusion-model)

<details><summary>Please consider cite <strong>AIST, AIST++, TalkSHOW, MotionDiffusion</strong> if they also help on your project</summary>

```bibtex

@inproceedings{aist-dance-db,
  author = {Shuhei Tsuchida and Satoru Fukayama and Masahiro Hamasaki and Masataka Goto}, 
  title = {AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing}, 
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR) },
  year = {2019}, 
  month = {Nov} 
}

@inproceedings{li2021learn,
  title={AI Choreographer: Music Conditioned 3D Dance Generation with AIST++}, 
  author={Ruilong Li and Shan Yang and David A. Ross and Angjoo Kanazawa},
  year={2021},
  booktitle={ICCV}
}

@inproceedings{yi2023generating,
  title={Generating Holistic 3D Human Motion from Speech},
  author={Yi, Hongwei and Liang, Hualin and Liu, Yifei and Cao, Qiong and Wen, Yandong and Bolkart, Timo and Tao, Dacheng and Black Michael J},
  booktitle={CVPR}, 
  pages={469-480},
  month={June}, 
  year={2023} 
}

@inproceedings{tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={ICLR},
  year={2023},
  url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}


```
</details>


 
# Usage 

### Training 

The results will be saved in $workspace. Please change it in the config/*.yaml files.
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

# Tips
* Using an appropriate learning rate for SMPL-X shape is important to learn accurate shape. 
* Omnidata normal supervision can effectively enhance the overall geometry and texture consistency; however, it demands more time for optimization.

# Other Interesting Works
* [HumanNorm](https://humannorm.github.io/): multiple stage SDS loss and perceptual loss can help generate the lifelike texture.
 
# Citation

```bibtex
@inproceedings{liao2024tada,
  title={{TADA! Text to Animatable Digital Avatars}},
  author={Liao, Tingting and Yi, Hongwei and Xiu, Yuliang and Tang, Jiaxiang and Huang, Yangyi and Thies, Justus and Black, Michael J.},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}
```

# License
This code and model are available for non-commercial scientific research purposes as defined in the LICENSE (i.e., MIT LICENSE). 
Note that, using TADA, you have to register SMPL-X and agree with the LICENSE of it, and it's not MIT LICENSE, you can check the LICENSE of SMPL-X from https://github.com/vchoutas/smplx/blob/main/LICENSE; Enjoy your journey of exploring more beautiful avatars in your own application.
