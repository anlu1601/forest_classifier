from clustering.functions import cluster_and_plot
from folders import get_tso, get_tsd
from gui.plot_image import plot_trees_and_select
from tree_segmentation.tree_predictor import TreePredictor
import labeling.cnn as cnn

if __name__ == '__main__':
    # Predict and save trees
    predictor = TreePredictor()
    predictor.load_model('trained_model.h5')

    predictor.predict_and_store_trees_from_folder(folder_path="./tree_segmentation/data", save_json=True, save_folder=get_tso())

    # Choose trees to cluster
    #x_1, x_2, y_1, y_2 = plot_trees_and_select(get_tso())

    # Cluster and show trees
    #cluster_and_plot(x_1, x_2, y_1, y_2)

    model = cnn.two_layer_cnn()
    model.train_and_eval()
    model.create_predictions()





