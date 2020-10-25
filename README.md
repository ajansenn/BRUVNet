# BRUVNet (Baited Remote Underwater Video Net)

STOP! Watch and Star this repository to stay up to date on its progress. Create issues for additional features you'd like to see go into the BRUVNet pipeline or problems that need to be solved for fish AI models. 

![bruvnet](https://github.com/ajansenn/BRUVNet/blob/master/BRUVNet%20Image.png)

BRUVNet, a collaboration between [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth) and the [Supervising Scientist](http://environment.gov.au/science/supervising-scientist), is an open-sourced dataset of freshwater and marine fish images used for fisheries monitoring and research. Labelled images and models can be used for object detection, image segmentation and computer vision objectives, automating fish identification and quantification processes.

It aims to consolidate labelled fish datasets into one gold standard allowing users to subset and query easily. This will enable scientists, environmental managers and academics to research and develop their own custom models for fish species identification.

Associated with BRUVNet is the workflow which enables the distribution of images for labelling to multiple users in a controlled and secured manner using Microsofts [VoTT (Visual Object Tagging Tool)](https://github.com/Microsoft/VoTT). 

BRUVNet holds image labelling challenges which calls on Citizen Scientists of the world to help build our dataset! If you'd like to contribute, access your discrete set of images to label by going to the image labelling challenge page on www.bruvnet.org


## Data Preparation

BRUV footage most commonly exists in continuous video format, therefore fisheries scientists are required to extract frames or still images from video that will be used for annotations. 

### Remove empty frames from your footage 

The extract-score-store jupyter notebook in the notebooks folder of this repository is used to loop through a folder of videos stored locally or in the cloud, extract frames at a desired frame/second rate, score against a compact or cloud hosted model, and return all the images that have fish present. This allows us to discard all the empty frames from a video and optimise the annotation process. Images are stored in Azure Blob containers which can be accessed by labelling tools using a SAS token. The current configuration creates new blob containers for each video to enable distribution of images across multiple users without overlap of labelling. 

Alternatively, if you have already extracted fish images from video, they can be uploaded to an Azure Blob container through the Azure Portal or Azure Storage Explorer, ready for annotation and labelling.


## How to label images of fish for segmentation

Instance segmentation requires images of fish to be labelled using polylines. We used Microsofts [VoTT (Visual Object Tagging Tool)](https://github.com/Microsoft/VoTT) to generate current annotations in the BRUVNet dataset. Polygons are traced around the fish by clicking to add vertices, tightly, around the outline of visible fish species. Left click on the mouse to drop the first vertice, then double click on the final vertice to close off the polygon. To speed up the labelling process and define what makes a "good" annotation for model training we developed the following criteria to standardise labelling across multiple users, projects and geographical regions.

1) **_Key features_**, fish were only labelled if key defining characteristics for species level identification were visible in the image, such as colouration or morphology. 

2) **_Orientation_**, fish directly facing toward or away from the camera were not labelled, as key features are often obscured making species level identification difficult. 

3) **_Depth_**, as fish move further away from the camera, ability to confidently identify reduces. Orientation combined with deteriorating light and turbidity makes species level identification difficult. Annotations were only made in clear conditions where the above criteria were also met.  

4) **_Obstruction_**, if a fish was obscured by debris, aquatic vegetation or other fish, masks are not overlapped or separated into two masks.  

![annotations](https://github.com/ajansenn/BRUVNet/blob/master/VoTT%20Fish%20Annotations.PNG)


## The taxonomy of labelling

A consistent naming convention is applied to all fish across freshwater and marine systems following the [The International Commission on Zoological Nomenclature](https://www.iczn.org/). Scientific names, in latin, are always used with a genus in capital letters followed by a species name (e.g. Amniataba percoides). If the species name is unknown, and there are multiple potential species, the abbreviated spp. is used (e.g. Amniataba spp.). If there is only one species and it is unnamed, the abbreviated sp. is used (e.g. Amniataba sp.).

The taxonomy tree below highlights an example of freshwater species currently in the BRUVNet dataset that have been taken to different levels of classification. Names highlighted in red indicate the lowest classification level possible from imagery i.e. we do not have 100% certainty of species classification without further investigation, often requiring manual handling of the specimen. 

![classification tree](https://github.com/ajansenn/BRUVNet/blob/master/Classification%20Tree1.jpg)

## Label format and metadata

Annotations and labels are in [COCO](https://cocodataset.org/#home) format. We provide a jupyter notebook in the notebooks folder, VOTT-COCO, to convert images labelled in VoTT to BRUVNet-COCO format. Attributions to contributors and associated metadata detailing the imagery location, collection time, habitat and wether it's from freshwater or marine systems is allocated to allow subsetting of the dataset for model training.

An example of metadata when uploading labelled datasets to BRUVNet Master. 

![metadata](https://github.com/ajansenn/BRUVNet/blob/master/Metadata.PNG)

### Visualisation of the BRUVNet metadata

Use the PowerBI package provided in this repository to connect to the BRUVNet_Master.json file and query the status of the dataset i.e. number of species, annotations and images.

![PowerBI](https://github.com/ajansenn/BRUVNet/blob/master/PowerBI%20Visual.PNG)

## AI Models

BRUVNet currently uses Azure Machine Learning service to train instance segmentation models with [Detectron2](https://github.com/facebookresearch/detectron2). 

An object detection model trained using bounding box annotations of freshwater fish species is available. The model was trainied using [CustomVision.ai](https://www.customvision.ai/) and exported as a compact tensorflow file. 


## Setting up your desktop

Follow the steps below to install this repository on your dekstop.

* Install [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/)
* Install [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2019.10-Windows-x86_64.exe)
  * Choose All Users
* Install [Git Desktop](https://desktop.github.com/)
* Open Git Desktop and clone https://github.com/ajansenn/BRUVNet.git
* Open Anaconda Prompt, cd to folder with cloned repository and [create conda environment] using the code below(https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local):
```
conda env create -f environment.yml
conda env list
conda activate bruvnet
conda install notebook ipykernel
ipython kernel install --user
```

Create a free [Microsoft Azure account](https://azure.microsoft.com/en-au/free/search/?&OCID=AID2100005_SEM_XxwBIQAABlbEr6x_:20200826223928:s&msclkid=2994d79425221578b0a388d30fcfa145&ef_id=XxwBIQAABlbEr6x_:20200826223928:s&dclid=CjgKEAjwkJj6BRDA4aKNxJ-T7AgSJABGqdLcustSw9LZ5QLQ1dADrXJugi-_KX713AHwKyZ1fyX9zvD_BwE) to gain access to all the tooling required.  

## Azure Machine Learning Service

```
Download or clone this repository
Launch Azure Studio
Create a new datastore that connects to an Azure Blob with images and labels
Create new dataset from datastore
Create new compute instance (GPU Standard NC6 with 56GB Ram is a good start)
In Notebooks, upload this entire repository folder
Open aml_detectron2_bruvcoco_local from the notebooks folder
Follow steps and adjust parameters to suit your needs
Train model
```

## Contribute! 

If you already have a labelled dataset of fish images BRUVNet encourages collaboration and integration of your data into the gold standard. Your contributions will further development of fish AI projects and ensure it is stored safely and securely through time. 

## Team

Steve van Bodegraven (Microsoft)

Andrew Jansen (Supervising Scientist)

Kris Bock (Microsoft)

Varma Gadhiraju (Microsoft)

Valeriia Savenko (Microsoft)

Andrew Esparon (Supervising Scientist)

