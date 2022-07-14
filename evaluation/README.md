## Environments
* Python - 3.7.10
* Pytorch - 1.7.0 
* CUDA - 11.2
* GPU - GeForce RTX 3090

We note that all the evaluation codes in this repository have been tested under the above environments.

## Preprocessing
To prepare for the validation dataset of ImageNet, you may simply follow the instructions as below:

1. Download the ImageNet ILSVRC2012 validation dataset (i.e., ILSVRC2012_img_val.tar).

2. Extract ILSVRC2012_img_val.tar to the val directory with
   ```
   mkdir val && tar -xvf ILSVRC2012_img_val.tar -C ./val
   ```

3. Copy preprocess_val_dataset.sh to the val directory and re-organize the validation images with
   ```
   bash preprocess_val_dataset.sh
   ```
   
4. Delete preprocess_val_dataset.sh in the val directory with
   ```
   rm preprocess_val_dataset.sh
   ```

5. Finally, the dataset folder should look like
   ```
   -- dataset
      -- train
      -- val
   ```

## Evaluate
Tips: For the pretrained models, you may find them in the following Google Drive link (https://drive.google.com/drive/folders/1f9zLDfee9nuC8pH_21JCd-pK1xeHUYEF?usp=sharing).

You may use the following command to reproduce the reported accuracy on your GPU:
```
python3 evaluate.py --model MODEL_NAME --dataset-root /PATH/TO/DATASET --device cuda --gpu-id 0
```
Alternatively, you may reproduce the reported accuracy on your CPU with the following command:
```
python3 evaluate.py --model MODEL_NAME --dataset-root /PATH/TO/DATASET --device cpu
```
Note that the supported models are summarized as follows:
```
[LightNet-20ms, LightNet-22ms, LightNet-24ms, LightNet-26ms, LightNet-28ms, LightNet-30ms] # w/o Squeeze-and-Excitation (SE) model

[LightNet-20ms-SE, LightNet-22ms-SE, LightNet-24ms-SE, LightNet-26ms-SE, LightNet-28ms-SE, LightNet-30ms-SE] # w/ Squeeze-and-Excitation (SE) module
```

Then, you are expected to get the following results

      LightNet-20ms:
        Evaluate Result: Total: 50000	Top1-Acc: 74.996	Top5-Acc: 92.170
      LightNet-22ms:
        Evaluate Result: Total: 50000	Top1-Acc: 75.230	Top5-Acc: 92.192
      LightNet-24ms:
        Evaluate Result: Total: 50000	Top1-Acc: 75.492	Top5-Acc: 92.342
      LightNet-26ms:
        Evaluate Result: Total: 50000	Top1-Acc: 75.870	Top5-Acc: 92.576
      LightNet-28ms:
        Evaluate Result: Total: 50000	Top1-Acc: 76.068	Top5-Acc: 92.674
      LightNet-30ms:
        Evaluate Result: Total: 50000	Top1-Acc: 76.392	Top5-Acc: 92.924
      LightNet-20ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 75.350	Top5-Acc: 92.250
      LightNet-22ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 76.066	Top5-Acc: 92.466
      LightNet-24ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 75.878	Top5-Acc: 92.574
      LightNet-26ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 76.306	Top5-Acc: 92.840
      LightNet-28ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 76.524	Top5-Acc: 92.846
      LightNet-30ms-SE:
        Evaluate Result: Total: 50000	Top1-Acc: 76.990	Top5-Acc: 93.106
