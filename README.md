# GoogleNet Model
GoogleNet is made of three files:  
  1.Divide_DataSet.py: divide data set into train ,validation, test set
  2.GN.py: GooglNet Model training and save model as inception_1.h5
  3.Load_Predict.py: Load and use .h5 file for prediction class of unknown trees.
 GoolgNet branch includes folders:
  1.after_divide_dataset: train, validation and test set are here
  2.dataset: original dataset, include birch folder, spruce floder..
  3.dataset-division: used folder, can be ignored
 Others:
  1.GoogleNet.png: This program's GoogleNet Structure, including 22 layers
  2.inception_1.h5 : trained model, can be called for prediction directly.
