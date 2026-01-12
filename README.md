<div align="center">
<h2>Beyond External Guidance: <br>Unleashing the Semantic Richness Inside Diffusion Transformers for Improved Training</h2>

<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


[Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2</sup>
| [Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN)<sup>1,2</sup> | 
[Zhengqiang Zhang](https://scholar.google.com/citations?user=UX26wSMAAAAJ&hl=en)<sup>1</sup> | 
[Ruibin Li](https://scholar.google.com/citations?user=FMNs2K0AAAAJ&hl=en)<sup>1</sup> | 
[Yujing Sun](https://scholar.google.com/citations?user=kj3VUSwAAAAJ&hl=en)<sup>1,2</sup> |
[Shuaizheng Liu](https://scholar.google.com/citations?user=wzdCc-QAAAAJ&hl=zh-CN)<sup>1,2</sup> |
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute
</div>


![Self-Transcendence](figs/vis_com_256.png)

## 🧡ྀི Summary
We answer this question: ***Can internal features be used as effective semantic guidance signals to improve the training of DiT models?*** and introduce *Self-Transcendence*, a simple yet effective self-guided training strategy achieving REPA-level performance without any external feature supervision. Our proposed approach produces more discriminative and semantically richer features, as pre-trained DINO used in [REPA](https://github.com/sihyun-yu/REPA). Our method significantly improves training efficiency and generation quality, *acheiving FID=1.25 at just 400 epochs*.

![Self-Transcendence](figs/intro.png)


## ⏰ Update
- **2026.1.12**: The paper and this repo are released.

:star: If Self-Transcendence is helpful to your images or projects, please help star this repo. Thanks! :hugs:



## 🌟 Overview
<div align="center">
<img src="figs/framework.png" height="480px"/>
</div>