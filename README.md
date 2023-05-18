# Deep Orthogonal Multi-Frequency Fusion for Tomogram-Free Diagnosis in Diffuse Optical Imaging
This is the software repository for [our](https://www2.cs.sfu.ca/~hamarneh/ecopy/techrxiv_21574533.pdf) [paper](#cite) solving problem of breast cancer lesion detection and discrimination in novel orthogonal multi-frequency fusion paradigms.
## Motivation
Identifying breast cancer lesions with a portable diffuse optical tomography (DOT) device can improves early detection, while avoiding otherwise unnecessarily invasive, ionizing, and more expensive modalities such as CT, as well as enabling pre-screening efficiency. 
To accurately capture the highly heterogeneous tissue of a cancer lesion embedded in healthy breast tissue with non-invasive DOT, multiple frequencies can be combined to optimize signal penetration and reduce sensitivity to noise.
We show that an orthogonal multi-frequency DOT fusion can improve reconstruction and leads to more accurate end-to-end identification of malignant from benign lesions and healthy tissue, illustrating its regularization properties on the multi-frequency input space. Furthermore, we investigates the merits of tackling the diagnosis prediction task from raw sensor data directly without image reconstruction in DOT (direct prediction) and highlighs the potential of the raw-to-task model for improved accuracy, while reducing computational complexity.
![Fig1_TMI2](Images/Fig1_TMI2.png)

## Citation
<a name="Cite"></a>
```bibtext
@article{ben2022orthogonal,
  title={Orthogonal Multi-frequency Fusion Based Image Reconstruction and Diagnosis in Diffuse Optical Tomography},
  author={Ben Yedder, Hanene and Cardoen, Ben and Shokoufi, Majid and Golnaraghi, Farid and Hamarneh, Ghassan},
  year={2022},
  publisher={TechRxiv}
}
```

## Table of contents
1. [Contribution](#contribution)
2. [Installation](#install)
3. [Usage](#usage)
4. [Questions?](#faq)

### Contribution
<a name="contribution"></a>
- We investigate the benefit of multi-frequency data on the quality of DOT reconstruction and breast lesion diagnosis using deep learning.
- We propose a novel approach designed to recover the optical properties of breast tissue from multi-frequency data with a deep orthogonal fusion model followed by a diagnosis.
- We investigate the merits of tackling the diagnosis prediction task from raw sensor data directly without image reconstruction in DOT (direct prediction).
-  We extend a fusion network by training models using an orthogonalization loss function to maximize the independent contribution of each modulation frequency data and
emphasize their collective strength. 
### Installation
<a name="install"></a>

```bash
git clone https://github.com/haneneby/FuseNet.git  
cd FuseNet
conda env create -f requirements.txt --name FuseNetest
conda activate FuseNetest
```
### Usage
<a name="usage"></a>
```bash
mkdir myoutput
python3 FuseNet++.py --epochs 10 --outputroot myoutput
```
This will train the network and save output in `myoutput`.
Examples of outputs are presented in [Images](FuseNet/Images) 
<!--![images/reconst](FuseNet/Images/test_generated_image-19.png?=100x100)-->


### Questions?
<a name="faq"></a>
Please create a [new issue](https://github.com/haneneby/FuseNet/issues/new/choose)  detailing concisely, yet complete what issue you encountered, in a reproducible way.

