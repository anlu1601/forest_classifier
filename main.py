from clustering.functions import cluster_and_plot
from folders import get_tso, get_tsd
from gui.plot_image import plot_trees_and_select
from tree_segmentation.tree_predictor import TreePredictor
import interaction.user_interaction as interact
import interaction.import_trees as imp_trees
import labeling.choose_model as cm
import clustering.make_cluster as mc




if __name__ == '__main__':
    # PREDICT AND SAVE TREES, SEGMENTATION
    #predictor = TreePredictor()
    #predictor.load_model('trained_model.h5')
    #predictor.predict_and_store_trees_from_folder(folder_path="./tree_segmentation/data", save_json=True, save_folder=get_tso())

    # CREATE CLUSTER AND SHOW USER
    data, forest_image = imp_trees.get_trees_from_random_forest()
    cluster = mc.make_cluster(forest_image)
    forest_cluster = interact.tree_data(data, cluster)
    forest_cluster.show_location()
    forest_cluster.show_forest_cluster()

    #forest_cluster.show_labeled('Spruce')
    #forest_cluster.set_label('TestLabel')

    # CREATE AND RUN CLASSIFICATION
    # Can be 'Inception' or 'cnn'
    cm.run_classifier('cnn')





