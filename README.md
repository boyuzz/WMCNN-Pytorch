# WMCNN-Pytorch
The Pytorch reproduction of WMCNN [Aerial Image Super Resolution via Wavelet Multiscale Convolutional Neural Networks]
If you use this code, please cite the paper.

****
The PSNR value on RSSCN7 dataset is compared in the following table.

|Methods|Upscaling factor|Grass|Field|Industry|River Lake|Forest|Resident|Parking|Average|
|---|---|---|---|---|---|---|---|---|---|
|WMCNN_paper|2|38.82|37.30|28.35|32.41|29.68|28.49|29.10|32.02|
|WMCNN_pytorch|2|38.98|37.38|28.28|32.31|29.71|28.33|30.00|32.14|


****
# Usage

## Generate data
First, you need to download the RSSCN7 dataset in [this site](https://www.dropbox.com/s/j80iv1a0mvhonsa/RSSCN7.zip?dl=0&file_subpath=%2FRSSCN7) and put it in the directory "data/rsscn7". Then you can either use the following two methods to generate the hdf5 dataset. (*Note: using other datasets is also possible.)

### Matlab
Use the code "generate_train.m" provided in folder 'matlab_generate_data' to generate hdf5 dataset.

### Python
If the matlab is not available, you can use python code "data_generator.py" to generate hdf5 dataset.

## Training
1. First, update the dataset path in config file 'configs/wmcnn.json'
```
 "train_data_loader": {
    "data_path": "./data/rsscn7/",
    "train_path": "full_wdata.h5",
    ...
    }
```
2. Use the following code for training,
```
python main_train.py -c configs/wmcnn.json
```
After training, the config file will be copied to 'experiments/wmcnn/wmcnn.json'
You can check the log during training in 'experiments/wmcnn/log' with tensorboard.

## Testing
1. First, update the testing dataset path in config file 'experiments/wmcnn/wmcnn.json'
```
  "test_data_loader": {
    "data_path": "./data/rsscn7/",
    "test_path": "Samples/",
    "upscale": 2
  }
```
2. Use the following code for testing,
```
python main_test.py -c experiments/wmcnn/wmcnn.json
```
