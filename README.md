# Deep Orthogonal Multi-Frequency Fusion for Tomogram-Free Diagnosis in Diffuse Optical Imaging
This is the software repository for [our](https://www2.cs.sfu.ca/~hamarneh/ecopy/techrxiv_21574533.pdf) [paper](#cite) solving problem of breast cancer lesion detection and discrimination in a novel way.
## Motivation
Identifying breast cancer lesions with a portable diffuse optical tomography (DOT) device can improves early detection, while avoiding otherwise unnecessarily invasive, ionizing, and more expensive modalities such as CT, as well as enabling \review{ pre-screening efficiency. 
To accurately capture the highly heterogeneous tissue of a cancer lesion embedded in healthy breast tissue with non-invasive DOT, multiple frequencies can be combined to optimize signal penetration and reduce sensitivity to noise.
We show that an orthogonal multi-frequency DOT fusion can improve reconstruction and leads to more accurate end-to-end identification of malignant from benign lesions and healthy tissue, illustrating its regularization properties on the multi-frequency input space. Furthermore, we investigates the merits of tackling the diagnosis prediction task from raw sensor data directly without image reconstruction in DOT (direct prediction) and highlighs the potential of the raw-to-task model for improved accuracy, while reducing computational complexity.

## Table of contents
1. [Installation](#install)
2. [Test](#test)
3. [Usage](#usage)
4. [Cite](#cite)
5. [Reproducing results](#repro)
   1. [Datasets](#data)
   2. [Preprocessing](#script)
6. [FAQ](#faq)

<a name="installation"></a>
### Installation
#### Local
```bash
git clone git@github.com:haneneby/FuseNet.git  
cd FuseNet
conda env create -f requirements.yml --name FuseNetest
conda activate FuseNetest
```
