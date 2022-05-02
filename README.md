## SHAPE: An Unified Approach to Evaluate the Contribution and Cooperation of Individual Modalities

This repository corresponds to the PyTorch implementation of our model for calculating SHAPE score.

### Setup
Install Python >= 3.5
Install Cuda >= 9.0 and cuDNN
Install PyTorch >= 0.4.1 with CUDA (Pytorch 1.x is also supported).

Install SpaCy and initialize the GloVe as follows:
``` python
$ pip install -r requirements.txt
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
$ pip install en_vectors_web_lg-2.1.0.tar.gz
``` 

### Dataset 
#### 1. CMU-MOSEI
This dataset is from the work `https://github.com/jbdel/MOSEI_UMONS`, who took more pre-precessing for the original dataset. You can Download it from [here](https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing)

#### 2. B-T4SA 
This dataset can be download from [here](http://t4sa.it/). 


#### 3. SNLI & VSNLI
This dataset can be download from [here](https://drive.google.com/file/d/15I553IyAua69V6F8jWGMT76OHMmXp0u8/view?usp=sharing)


#### Processed Dataset
We also provide the processed data to help quickly implement our code. The processed data can be download as follows:
1. [Baidu Driver](https://pan.baidu.com/s/1ap0pHl_UqA1gqw4Pw-XWLA). Code:`qnwy` 
2. [Google Driver](https://drive.google.com/file/d/1P-UtneDqTlU2S0iwPSc08bKKBTDzzNBe/view?usp=sharing)




Finally, the `datasets` folders will have the following structure

```angular2html
|-- datasets
	|-- MOSEI
        |  |-- train_mels.p
        |  |-- train_sentences.p
        |  |-- train_sentiment.p
        |  |-- valid_mels.p
        |  |-- valid_sentences.p
        |  |-- valid_sentiment.p
        |  |-- test_mels.p
        |  |-- test_sentences.p
        |  |-- test_sentiment.p
        |  |-- key_list (Processed Data)

    |-- T4SA
        |  |-- b-t4sa_train.txt
        |  |-- b-t4sa_val.txt
        |  |-- b-t4sa_test.txt
        |  |-- data (images)
        |  |-- raw_tweets_text.csv
        |  |-- t4sa_text_sentiment.tsv
        |  |-- data_pro (Processed Data)
    |-- SNLI 
        |  |-- flickr30 (images)
        |  |-- vsnli
        |  |  |-- VSNLI_1.0_train.tsv
        |  |  |-- VSNLI_1.0_dev.tsv
        |  |  |-- VSNLI_1.0_test_hard.tsv
        |  |  |-- VSNLI_1.0_test.tsv
        |  |-- token (Processed Data)

```


### Training 
The following script will start training with the default hyperparameters:

```bash
$ python train.py --model= "Dynamic" --dataset= "T4sa" --root_dir="/xx"
```
1. ```--model=str```, e.g. Dynamic', 'Dense', 'Late','Early'
2. ```--dataset=str```, e.g. 'T4sa', 'Mosei', 'SNLI'

Other setting can be found in the `trian.py` script. 


### Evaluate & Shape Score
Here we provede two scripts `evaluate.py` and `evaluate_perceptual.py`, which can be used to calculate the sub-factor of SHAPE Score. 

You can get `accuracy' and the 'accuracy' of zero-padded input for a specific modality through:
```bash
$ python evaluate.py --model="Dynamic" --dataset= "T4sa"
```

Then you can calculate the finaly Shape score through those ouput `accuracy`, following the description of our paper. 


To get  `in-class` and `out-class` version of the `Perceptual Score`:
```bash
$ python evaluate_perceptual --model="Dynamic" --dataset="T4sa" --average_number=10 
```


### Source Code Reference 
1. https://github.com/jbdel/MOSEI_UMONS
2. https://github.com/MILVLG/mcan-vqa
### Citation
