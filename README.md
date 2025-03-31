
## Pretrained ML Models & Datasets with Aerial/Drone Imagery in Mind


### Deepforest
[Deep Forest](https://deepforest.readthedocs.io/en/v1.5.0/index.html) is a python library built on top of Pytorch that does object detection from high-resolution aerial imagery

* Includes pretrained models detect tree crowns, identify birds, identify cattle, and detect alive v. dead trees.

* The tree crown model started with resnet-50 classification backbone pretrained on [ImageNet dataset](https://www.image-net.org/index.php). They further trained the model on these unsupervised lidar tree crowns (30 million) derived from NEON Lidar data across the US. Then final step was training on 10,000 hand-annoted canopy bounding boxes.  

* Example Code in a [Jupyter Notebook](https://github.com/ua-datalab/Geospatial_Workshops/wiki/Image-Object-Detection-%E2%80%90-Deep-Forest)

<br>
<br>

### Restor Tree Crown Delineation
[Restor Tree Crown Delineation](https://restor-foundation.github.io/tcd/): Consists of a [tree crown dataset](https://huggingface.co/datasets/restor/tcd), and [pretrained models](https://huggingface.co/restor) that do semantic segmentation (segformer & unet) and instance segmentation (mask-rcnn). Also has a coded pipeline to train and predict tree crowns. 

<br>
<br>

### Detectree2

[Detectree2](https://patball1.github.io/detectree2/) is a python library that consists of [pretrained models](https://zenodo.org/records/15014353) that do instance segmentation of individual tree crowns from aerial and drone imagery. 

<br>
<br>


### Tree Detection Framework from Open Forest Observatory
[Tree Detection Framework](https://github.com/open-forest-observatory/tree-detection-framework)

<br>
<br>

### Detecto
[Detecto](https://detecto.readthedocs.io/en/latest/) is an object detection python library built on top of deep learning framework [Pytorch](https://pytorch.org/). 

By default, Detecto uses the convolutional neural network architecture [Faster R-CNN](https://arxiv.org/pdf/1506.01497) ResNet-50 FPN. The architecture was pre-trained on the [COCO](https://cocodataset.org/#home) (common objects in context) dataset which contains over 330,000 images, each annotated with 80 object classes (e.g., person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, kite, knife, spoon, tv, book). The images and labels are generally not from aerial view points. Therefore, Detecto is not ready to identify objects in aerial images out-of-the-box. It has to be trained to do so.

[Lesson using Detecto](https://github.com/ua-datalab/Geospatial_Workshops/wiki/Image-Object-Detection-%E2%80%90-Detecto) to train and identify objects from aerial imagery. It uses the [NWPU VHR-10 dataset](https://github.com/chaozhong2010/VHR-10_dataset_coco) which is a very high resolution (VHR) aerial imagery dataset that consists of 800 total images. The dataset has has ten classes of labeled objects: 1. airplane(757), 2. ship(302), 3. storage tank(655), 4. baseball diamond(390), 5. tennis court(524), 6. basketball court(159), 7. ground track field(163), 8. harbor(224), 9. bridges(124), and 10. vehicle(477). 

[Example code in a Jupyter Notebook to identify lettuce](https://github.com/jeffgillan/data_to_science_scripts/blob/main/lettuce_detecto.ipynb)


<br>
<br>

### WALDO
[W.A.L.D.O.](https://huggingface.co/StephanST/WALDO30) is a pretrained model for detecting objects in aerial video or images. 

The model can identity 12 object classes: 0 = light vehicle; 1 = person; 2 = building; 3 = Utility pole; 4 = boat; 5 = bike; 6 = container; 7 = truck; 8 = gastank; 10 = digger (construction equipment); 11 = solar panels; 12 = bus. The training dataset is not public. 

WALDO pretrained model is based on [YOLO-v8](https://docs.ultralytics.com/models/yolov8/) which was trained on the [COCO](https://cocodataset.org/#home) dataset. 

Go [here](https://github.com/jeffgillan/yolo_drone) for example code on how to detect objects in mp4 drone videos. 

<br>
<br>

### VisDrone Dataset
[VisDrone from Ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/visdrone/)
The VisDrone Dataset is a large-scale benchmark created by the AISKYEYE team at Tianjin University, China. It is designed for various computer vision tasks related to drone-based image and video analysis. 

You can fine-tune the yolo vision model on the dataset for various tasks like object detection, tracking, and crowd counting. 

<br>
<br>

### Low Altitude Disaster Imagery
[Low Altitude Disaster Imagery (LADI)](https://github.com/LADI-Dataset/ladi-overview) comprises of datasets and pretrained models to identify things after a disaster like damaged buildings and trees. The pretrained models are multi-label classifications meaning they will look at an image and output things like 'building_damage = true', 'tree_damage = false'. It does not do object detection or segmentation.  

The [LADI dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset) comprises of 10k aerial images collected and labeled by the Civil Air Patrol of natural disasters. 

LADI provdies two pretrained models: [LADI-v2-classifier-small](https://huggingface.co/MITLL/LADI-v2-classifier-small) and [
LADI-v2-classifier-large](https://huggingface.co/MITLL/LADI-v2-classifier-large). The small model was trained from [google/bit-50](https://huggingface.co/google/bit-50) - image classification model pretrained on datasets such as Image-Net-21k and JFT-300m. The large model was trained from [Swin Transformer v2](https://huggingface.co/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft), which is a transformer architecture trained on ImageNet data. 

<br>
<br>

### GeoAI python library from Quisheng Wu
[GeoAI](https://geoai.gishub.org/)

[SAMGeo](https://samgeo.gishub.org/)


<br>
<br>



### TorchGeo NAIP Foundation Model
[TorchGeo NAIP Foundation Models](https://torchgeo.readthedocs.io/en/stable/api/models.html#naip)

[satlas_pretrain models](https://github.com/allenai/satlaspretrain_models/)

These are transformer-based 'foundation' models, meaning original training is self-supervised on large datasets of NAIP aerial imagery. They have not been trained to do a specific task (e.g., object detection) but can be trained to do so. 

The following prediction head architectures are available:

Segmentation: U-Net Decoder w/ Cross Entropy loss
Detection: Faster R-CNN Decoder
Instance Segmentation: Mask R-CNN Decoder
Regression: U-Net Decoder w/ L1 loss
Classification: Pooling + Linear layers
Multi-label Classification: Pooling + Linear layers

<br>
<br>


### Dota Dataset
A Large-Scale Benchmark and Challenges for Object Detection in Aerial Images

[DOTA Dataset](https://captain-whu.github.io/DOTA/index.html)

<br>
<br>

### SODA-A Dataset
https://huggingface.co/datasets/satellite-image-deep-learning/SODA-A

SODA-A comprises 2513 high-resolution images of aerial scenes, which has 872069 instances annotated with oriented rectangle box annotations over 9 classes.


<br>
<br>

## Drone Photogrammetry Processing 

[Automated Metashape workflow on local machine](https://github.com/jeffgillan/automate-metashape), native & docker

[Automated Metashape worfklow on HPC](https://github.com/jeffgillan/metashape_hpc)

[OpenDroneMap](https://github.com/jeffgillan/opendronemap)

[Convert geotiff to COGs and point clouds to COPC (docker)](https://github.com/jeffgillan/cog_copc_generate)

[Pipeline for automated Metashape workflow + COG/COPC conversion](https://github.com/jeffgillan/cog_copc_generate), docker-compose

<br>
<br>

## Cyber Infrastructure

[Open Forest Observatory](https://openforestobservatory.org/)

[CACAO Open Forest Observatory](https://github.com/open-forest-observatory/cacao-terraform-ofo/tree/main)  

[STAC catalog for Open Forest Observatory](https://github.com/open-forest-observatory/stac)

[Data to Science](https://ps2.d2s.org/)

Script to upload imagery products to Data-to-Science


