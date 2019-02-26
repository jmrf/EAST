# EAST: An Efficient and Accurate Scene Text Detector

### Introduction

This is mainly a fork repackaging [argman/EAST](https://github.com/argman/EAST) code-base making it more accessible as a package to other pieces of codes to import and experiment.

---

This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).

The features are summarized blow:

-   [Online demo](http://east.zxytim.com/) + [Result example](http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e)
    CAVEAT: There's only one cpu core on the demo server.
    Simultaneous access will degrade response time.
-   Only **RBOX** part is implemented.
-   A fast Locality-Aware NMS in C++ provided by the paper's author.
-   The pre-trained model provided achieves **80.83** F1-score on ICDAR 2015 (Incidental Scene Text Detection Challenge) using only training images from ICDAR 2015 and 2013. See [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_samples&task=1&m=29855&gtv=1) for the detailed results.
-   Differences from original paper:
    -   Use ResNet-50 rather than PVANET
    -   Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
    -   Use linear learning rate decay rather than staged learning rate decay
-   Speed on 720p (resolution of 1280x720) images:
    -   Now
        -   Graphic card: GTX 1080 Ti + Network fprop: **~50 ms**
        -   NMS (C++): **~6ms**
        -   Overall: **~16 fps**
    -   Then
        -   Graphic card: K40 + Network fprop: ~150 ms
        -   NMS (python): ~300ms + Overall: ~2 fps

Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please cite his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents

1. [Installation](#installation)
2. [Download](#download)
3. [Demo](#demo)
4. [Test](#train)
5. [Train](#test)
6. [Examples](#examples)

### Installation

The Makefile `east/lanms/` has been modified to avoid issues with gcc < 6 and **-fno-plt** flag including hardcoded paths.
If you find the same issue, please do the following:

```bash
    python3-config --cflags
```

Copy the output, remove `-fno-plt` and paste in the Makefile as:
`CXXFLAGS = <output-from-previous-command>`

Then:

```bash
cd east/lanms
make
```

Finally:

```bash
    sudo apt-get install libgeos-dev  # required for Shapely
    pip install -r requirements.txt
```

### Download

1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

To download the models automatically:

```bash
    bash download_models.sh
```

### Train

If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```bash
python -m east.multigpu_train \
    --gpu_list=0 \
    --input_size=512 \
    --batch_size_per_gpu=14 \
    --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
    --text_scale=512 --training_data_path=/data/ocr/icdar2015/ \
    --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
    --pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note: you should change the gt text file of icdar2015's filename to img*\*.txt instead of gt_img*\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Demo

If you've downloaded the pre-trained model, you can setup a demo server by

```bash
python east.run_demo_server \
    --checkpoint-path /checkpoints/east_icdar2015_resnet_v1_50_rbox/
```

Then open http://localhost:8769 for the web demo. Notice that the URL will change after you submitted an image.
Something like `?r=49647854-7ac2-11e7-8bb7-80000210fe80` appends and that makes the URL persistent.
As long as you are not deleting data in `static/results`, you can share your results using the same URL.

URL for example below: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
![web-demo](demo_images/web-demo.png)

### Test

run

```bash
python eval.py \
    --test_data_path=/tmp/images/ \
    --gpu_list=0 \
    --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
    --output_dir=/tmp/
```

a text file will be then written to the output path.

### Examples

Here are some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)

### Troubleshooting

-   How to compile lanms on Windows ?
    -   See https://github.com/argman/EAST/issues/120

Please let me know if you encounter any issues(my email boostczc@gmail dot com).
