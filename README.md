# forest_classifier
forest_classifier is a python application that is capable to repeatedly create labels for tree species in a forest with help of machine learning and human interaction, based on aerial imagery. 
These labels can then be used to predict the unlabeled trees. forest_classifier takes aerial images all the way to individual tree species predictions.

## Installation
forest_classifier is only available to install from source using the github repository. 
The python package dependencies are managed by conda. For help installing conda see: [conda quickstart](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```bash
git clone https://github.com/anlu1601/forest_classifier.git
cd forest_classifier
conda env create --file=environment.yml
conda activate forest_classifier
#build c extentions for retinanet
cd ./tree_segmentation
python setup.py build_ext --inplace
```



## Usage
The main.py file can be run as is after forest_classifier is installed. Place aerial forest images in *./tree_segmentation/data*. Main.py will then generate individual tree images.
It will then generate a cluster from a random aerial image and show this cluster for it to be determined a label. This process is meant to be repeating.
Then based on the detemined labels a classification algorithm will predict the species of the non labeled tree images and write them to *prediction_list.csv*

### Prediction
The prediction of tree crowns is made using the [DeepForest](https://github.com/weecology/DeepForest) package which is included from source in forest_classifier.
This prediction needs a prebuilt model to be loaded. The prediction is made so it works with many images.
Just input however many images to folder_path and all of them will be segmentated into tree images and relevant meta data will be saved as a.json file in same directory.
For more information on how to train a model and use it, see [DeepForest](https://github.com/weecology/DeepForest).

```python
from folders import get_tso, get_tsd
from tree_segmentation.tree_predictor import TreePredictor

# Create object
predictor = TreePredictor()

# Load existing model
predictor.load_model('trained_model.h5')

# Predict trees and save them
predictor.predict_and_store_trees_from_folder(folder_path=folder_path, save_json=True, save_folder=get_tso())
```

### Clustering
To create a cluster there is three pre-existing models to choose from but by default it is a implementation of INCEPTION_3.
This model extracts the features and then two random tree images is chosen to be the cluster origins.
The rest of tree images is compared to these origins and goes to the most similar cluster.

```python
import interaction.import_trees as imp_trees
import labeling.choose_model as cm

# Load data
data, forest_image = imp_trees.get_trees_from_random_forest()

# Create cluster
cluster = mc.make_cluster(forest_image)
```

### User interaction
The user interaction comes in form of simple function calls. The object takes tree data and a cluster and it shows to the user the location of the image from meta data.
It also shows the tree images from a cluster for easier labeling process. Then the functions exists for user to set label for all trees in the cluster.
Also for showing all trees with a specific label. Also exists function for easy deletion incase of wrongly determined label.


```python
import interaction.user_interaction as interact
import interaction.import_trees as imp_trees

# Load data
data, forest_image = imp_trees.get_trees_from_random_forest()

# Create object
forest_cluster = interact.tree_data(data, cluster)

# Example function calls
forest_cluster.show_location()
forest_cluster.show_forest_cluster()
forest_cluster.show_labeled('Spruce')
forest_cluster.set_label('Pine')
forest_cluster.delete_labeled('Pine', 'tree_001.png')
```

### Classification
Clasification of unlabeled images from *./tree_segmentation/output* directory. Two algorithms exists for this classification. 
First a basic two-layered cnn simply called cnn and then an implementation of GoogleNet/ InceptionV1. *run_classifier* loads the data from the directory, preprocess it and does the training and validation.
GoogleNet also does testing. Then a prediction list is generated for all trees.

```python
import clustering.make_cluster as mc

# Can be 'Inception' or 'cnn'
cm.run_classifier('cnn')
```



