Patch classification network

Requirements
- Python 3.7.16
- PyTorch 1.6.0
- Matplotlib 3.5.3

setup

conda create --name patch_classfication python=3.7 -y
conda activate patch_classfication
cd [your path]/Patch_classfication_network           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies



Datasets
- We have referred to the format of the VOC dataset for the format of the training dataset. As shown in the example below, we can place custom datasets in the dataset folder, provided that the dataset is created in VOC format and placed along the correct path.
- It is worth noting that we need to rename the images and their corresponding segmentation labels placed in the JPEGImages and SegmentationClass file directories to (number)_ 0 or (number)_ 1 format. Among them, (number) represents the number of the image. Each numbered image should consist of an original image and its corresponding label.The suffix 0 represents an image with negative samples or no defect areas, while the suffix 1 represents an image with positive samples or defect areas. For example, if there is no wire fracture in the 203th image, we will name its original image and segmentation label 203_ 0.jpg and 203_ 0. png.

- Patch Classfication Network
    - dataset
    +---VOCDataset
    +------VOC2007
    +---------ImageSets
    +---------JPEGImages
    +---------SegmentationClass



How to Train
We can use the train.py file for network training
	1. We need to modify the folder path to be trained (in line 57 of the train.py file, the parameter name is VOCdevkit_path)
	2. We need to modify the save path for the training information (parameter name save_dir in line 53 of the train.py file)
	3. Based on the configuration of the computer equipment, we can adjust the batch during network training_ Size (in line 42 of train.py, the parameter name is Unfreeze_batch_size)
	4. We can modify the size of the input image during training as needed (in line 34 of train.py, the parameter name is input_shape)
	5. We can modify the initial learning rate (in line 45 of train.py, the parameter name is Init_lr)

Started training
	1 Simply input the training instructions at the terminal.
	- example : Python train.py

How to evaluate
We can use the test.py file to test the weights of network training
	1. We need to modify the location of the original image to be tested (on line 68 of the test. py file)
	-example : path = 'VOCdevkit_DXDG_2.0_/VOC2007/JPEGImages'
	2. We need to modify the directory of the files to be tested (on line 69 of the test. py file)
	-example : test_file_path = 'VOCdevkit_DXDG_2.0_/VOC2007/ImageSets/Segmentation/test.txt'
	3. We need to modify the path name for storing the output results (in line 70 of the test. py file)
	- example : metrics_out_path = "eval_dxdg/segdec"
	4. We need to modify it to the weight file that needs to be loaded by the network (on line 19 of the network.py file)
	- example : "model_path" : r'model_data/best_epoch_weights.pth'

test
	1. Just input the command at the terminal.
	- example : python test.py