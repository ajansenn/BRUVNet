# BRUVNet (Baited Remote Underwater Video Net)

![bruvnet](https://github.com/ajansenn/BRUVNet/blob/master/BRUVNet%20Image.png)

BRUVNet is a collaboration between [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth) and the [Supervising Scientist](http://environment.gov.au/science/supervising-scientist), which hosts and shares labelled images of fish used for computer vision and deep learning.   

It aims to consolidate labelled fish datasets into one gold standard, with associated metadata, allowing users to subset and query easily. This will enable scientists, environmental managers and academics to research and develop their own custom models for fish species identification.

Associated with BRUVNet is the workflow which enables the distribution of images for labelling to Citizen Scientists in a controlled and secured manner using Microsofts [VoTT (Visual Object Tagging Tool)](https://github.com/Microsoft/VoTT). 

BRUVNet holds image labelling challenges which calls on Citizen Scientists of the world to help build our dataset! If you'd like to contribute, access your discrete set of images to label by going to the image labelling challenge page on www.bruvnet.org


## Data Preparation

BRUV footage most commonly exists in video format, therefore fisheries scientists are required to extract frames or still images from video that will be used for annotations. The extract-score-store jupyter notebook in the notebooks folder of this repository is used to loop through a folder of videos stored locally, extract frames at a desired frame/second rate, score against the KakaduFish compact model provided in the models folder, and return all the images that have fish present. This allows us to discard all the empty frames from a video and optimise the annotation process. Images are stored in Azure Blob containers which can be accessed by labelling tools using a SAS token. 


## How to label images of fish for segmentation

Instance segmentation requires images of fish to be labelled using polylines. We used Microsofts [VoTT (Visual Object Tagging Tool)](https://github.com/Microsoft/VoTT) to generate the current annotations in th BRUVNet dataset. Polygons are traced around the fish by clicking to add vertices, tightly, around the outline of visible fish species. You click to drop the first vertice, then double click on the final vertice to close off the polygon. To speed up the labelling process and define what makes a "good" annotation for model training we developed the following criteria to standardise labelling across multiple users, projects and geographical regions.

1) **_Key features_**, fish were only labelled if key defining characteristics for species level identification were visible in the image, such as colouration or morphology. 

2) **_Orientation_**, fish directly facing toward or away from the camera were not labelled, as key features are often obscured making species level identification difficult. 

3) **_Depth_**, as fish move further away from the camera, ability to confidently identify reduces. Orientation combined with deteriorating light and turbidity makes species level identification difficult. Annotations were only made in clear conditions where the above criteria were also met.  

4) **_Obstruction_**, if a fish was obscured by debris, aquatic vegetation or other fish, masks were not overlapped.  

![annotations](https://github.com/ajansenn/BRUVNet/blob/master/VoTT%20Fish%20Annotations.PNG)


## The taxonomy of labelling

A consistent naming convention is applied to all fish across freshwater and marine systems. Scientific names, in latin, are always used with a genus in capital letters followed by a species name (e.g. Amniataba percoides). If the species name is unknown, and there are multiple potential species, the abbreviated spp. is used (e.g. Amniataba spp.). If there is only one species and it is unnamed, the abbreviated sp. is used (e.g. Amniataba sp.).

The taxonomy tree below highlights an example of freshwater species currently in the BRUVNet dataset that have been taken to different levels of classification. Names highlighted in red indicate the lowest classification level possible from imagery i.e. there is not 100% certainty of species classification without further investigation, often needed through manual handling of the specimen. 

![classification tree](https://github.com/ajansenn/BRUVNet/blob/master/Classification%20Tree1.jpg)

## Label format

Annotations and labels are in [COCO](https://cocodataset.org/#home) format. We provide a jupyter notebook in the notebooks folder, VOTT-COCO, to convert images labelled in VoTT to COCO format. 

## AI Models

BRUVNet currently uses Azure Machine Learning service to train instance segmentation models with [Detectron2](https://github.com/facebookresearch/detectron2). 

