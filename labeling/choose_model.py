import labeling.cnn as cnn
import labeling.GN_driver as gn

def run_classifier(Model):

    if(Model == 'Inception'):

        model = './labeling/inception_1.h5'
        to_be_predict_folder = './tree_segmentation/output'
        # step1: create trainset, validationset, testingset
        (train_root, validation_root, test_root) = gn.divideDataset(r".\labeling\input", r".\labeling\divided_output")
        # step2: build and predict
        (name, results) = gn.check_model(to_be_predict_folder, model, train_root, validation_root, test_root)
        # print(name)
        # print(results)
        return

    if(Model == 'cnn'):

        # CLASSIFICATION & PREDICTION OF SPECIES

        model = cnn.two_layer_cnn()
        model.train_and_eval()
        model.create_predictions()

        return

    print("Error, model does not exist")
    return 1