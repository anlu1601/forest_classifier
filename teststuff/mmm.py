import shutil
import os

pathsss = r"C:/Users/André/Desktop/master_forest_clas/forest_classifier/teststuff/output/"
ppp = r"C:/Users/André/Desktop/master_forest_clas/forest_classifier/teststuff/imgs/tree_0.png"
shutil.copy2(ppp, pathsss + "cluster0/tree_0.png")

shutil.copy2('./imgs/tree_0.png', './output/cluster1/')