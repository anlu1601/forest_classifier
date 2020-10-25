from clustering.functions import cluster_and_plot
from folders import get_tso, get_tsd
from gui.plot_image import plot_trees_and_select
from tree_segmentation.tree_predictor import TreePredictor
import interaction.user_interaction as interact
import interaction.import_trees as imp_trees
import labeling.choose_model as cm


if __name__ == '__main__':
    # PREDICT AND SAVE TREES, SEGMENTATION
    #predictor = TreePredictor()
    #predictor.load_model('trained_model.h5')
    #predictor.predict_and_store_trees_from_folder(folder_path="./tree_segmentation/data", save_json=True, save_folder=get_tso())

    # Choose trees to cluster
    #x_1, x_2, y_1, y_2 = plot_trees_and_select(get_tso())

    # Cluster and show trees
    #cluster_and_plot(x_1, x_2, y_1, y_2)
    #data = imp_trees.get_trees_from_random_forest()
    #forest_cluster = interact.tree_data(data)
    #forest_cluster.show_location()
    #forest_cluster.show_forest_cluster()
    #forest_cluster.set_label('TestLabel')

    # Can be 'Inception' or 'cnn'
    cm.run_classifier('Inception')





