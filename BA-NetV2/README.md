BA-NetV2 MMSegmentation
The implementation of BA-NetV2 is based on MobileNetV2 implemented by MMSegmentation. You can add it to the specified version of MMSegmentation and train it.

Model details
1. basic settings
    - backbone : BA-Netv2 backbone (base on MobilenetV2)
    - decode head : BA-Head

2. parameters and resluts:
    We change MobilenetV2 to BA-NetV2 Style, some configurations differ from the original version of MMSegmentation are as follows :
    - backbone :
	out_indices=(0, 1, 2)
	strides=(1, 2, 2)
	arch_settings = [[1, 24, 1], [6, 32, 2], [6, 48, 3]]
	dilations=(1, 1, 1)
    - decode head :
	in_channels = [24,32,48]
	base_channel = 24

please refer the config file config\banetv2_config.py for more details.


How To Use?

Please refer get_started.md for the env installation and basic usage of MMSegmentation. The env details as follows:

Ubuntu 18.04
Python: 3.8.16 
PyTorch: 2.0.0 + TorchVision: 0.15.0
NVCC: Cuda compilation tools, release 11.3, V11.3.109
GCC: 9.4.0
OpenCV: 4.7.0.72
MMCV: 1.7.1
MMSegmentation: 0.30.0+83e7cc2

3. clone our repo to your workstation

4. copy follwing files to the directory of mmdetection project
cd BANet-MMSegmentation
mv ./config/banetv2 ${your path to mmdetection}/configs/
mv ./config/_base_/datasets/cityscapes_.py ${your path to mmdetection}/configs/_base_/datasets/
mv ./mmseg/models/backbone/mobilenet_v2.py ${your path to mmdetection}/mmseg/models/backbone/
mv ./mmseg/models/decode_heads/banet_base.py ${your path to mmdetection}/models/decode_heads/

5. register and import module in __init__.py

mmseg/models/decode_heads/__init__.py

...
from .banet_base import BaHeadBase

__all__ = [
    ..., 'BaHeadBase'
]



Train and Test
6. prepare cityscapes type dataset
cd ${your path to mmsegmentation}
mkdir data && cd data
ln -s ${your path to cityscapes type dataset} ./

7. train
# single-gpu
cd ${your path to mmsegmentation}/tools
python ./train.py  ../config/banetv2/banetv2_config.py [optional arguments]
# multi-gpu
./dist_train.sh ../config/banetv2/banetv2_config.py ${GPU_NUM} [optional arguments]

8. test
# single-gpu
cd ${your path to mmsegmentation}/tools
python ./test.py  ../config/banetv2/banetv2_config.py ${CHECKPOINT_FILE} [optional arguments]
# multi-gpu
./dist_test.sh ../config/banetv2/banetv2_config.py ${CHECKPOINT_FILE} ${GPU_NUM} --out ${RESULT_FILE} [optional arguments]



