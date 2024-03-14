
This repository provides solution methods for the [Second REACT Challenge](https://sites.google.com/cam.ac.uk/react2024)

### Baseline paper:
https://arxiv.org/pdf/2401.05166.pdf



https://github.com/reactmultimodalchallenge/baseline_react2023/assets/35754447/8c7e7f92-d991-4741-80ec-a5112532460b

## 🛠️ Installation

### Basic requirements

- Python 3.8+ 
- PyTorch 1.9+
- CUDA 11.1+ 

### Install Python dependencies (all included in 'requirements.txt')

```shell
conda create -n react python=3.8
conda activate react
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description:**
- The REACT 2023 Multimodal Challenge Dataset is a compilation of recordings from the following three publicly available datasets for studying dyadic interactions: [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780), [RECOLA](https://ieeexplore.ieee.org/document/6553805) and [UDIVA](https://www.computer.org/csdl/proceedings-article/wacvw/2021/196700a001/1sZ3sn1GBxe). 

- Participants can apply for the data at our [Homepage](https://sites.google.com/cam.ac.uk/react2023/home).
   
**Data organization (`data/`) is listed below:**
```data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files```
The example of data structure.
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.png
                       ├── ....
                       ├── 751.png
                   ├── ....
           ├── ....
       ├── RECOLA
       ├── UDIVA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
            
```
 
- The task is to predict one role's reaction ('Expert' or 'Novice',  'P25' or 'P26'....) to the other ('Novice' or 'Expert',  'P26' or 'P25'....).
- 3D_FV_files involve extracted 3DMM coefficients (including expression (52 dim), angle (3 dim) and translation (3 dim) coefficients.
- The frame rate of processed videos in each site is 25 (fps = 25), height = 256, width = 256. And each video clip has 751 frames (about 30s), The samping rate of audio files is 44100. 
- The csv files for baseline training and validation dataloader are now avaliable at 'data/train.csv' and 'data/val.csv'
 
 
</p>
</details>



<details><summary> <b> Training </b>  </summary>
<p>
 
 <b>Trans-VAE</b>
- Running the following shell can start training Trans-VAE baseline:
 ```shell
 python train.py --batch-size 4  --gpu-ids 0  -lr 0.00001  --kl-p 0.00001 -e 50  -j 12  --outdir results/train_offline 
 ```
 &nbsp; or 
 
  ```shell
 python train.py --batch-size 4  --gpu-ids 0  -lr 0.00001  --kl-p 0.00001 -e 50  -j 12 --online  --window-size 16 --outdir results/train_online  
 ```
 
 <b>BeLFusion</b>
 - First train the variational autoencoder (VAE):
```shell
python train_belfusion.py config=config/1_belfusion_vae.yaml name=All_VAEv2_W50
```
 
 - Once finished, you will be able to train the offline/online variants of BeLFusion with the desired value for k:
```shell
python train_belfusion.py config=config/2_belfusion_ldm.yaml name=<NAME> arch.args.k=<INT (1 or 10)> arch.args.online=<BOOL>
```

 
</p>
</details>

<details><summary> <b> Pretrained weights </b>  </summary>
 If you would rather skip training, download the following checkpoints and put them inside the folder './results'.
<p>
 
 <b>Trans-VAE</b>: [download](https://drive.google.com/drive/folders/1tyLQnQj1e2SMArBkc3gHDZVHwSr_GEod?usp=share_link)
 
 <b>BeLFusion</b>: [download](https://ubarcelona-my.sharepoint.com/:f:/g/personal/germanbarquero_ub_edu/EvF9K27g_DFPp2MS_8OqkmwBYGzUKs7J3QmkidbRLVSt6Q?e=WCJ2JU)
 
</details>

<details><summary> <b> Validation </b>  </summary>
<p>
 Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.
 
- Before validation, run the following script to get the martix (defining appropriate neighbours in val set):
 ```shell
 cd tool
 python matrix_split.py --dataset-path ./data --partition val
 ```
&nbsp;  Please put files (`data_indices.csv`, `Approprirate_facial_reaction.npy` and `val.csv`) in the folder `./data/`.
  
- Then, evaluate a trained model on val set and run:

 ```shell
python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/val_offline --split val
```
 
&nbsp; or
 
```shell
python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/val_online --split val
```
 
- For computing FID (FRRea), run the following script:

```
python -m pytorch_fid  ./results/val_offline/fid/real  ./results/val_offline/fid/fake
```
</p>
</details>



