import json
import os

from tree_segmentation import paths
from deepforest import deepforest, get_data
from matplotlib import pyplot

from PIL import Image


class TreePredictor:

    def __init__(self):
        self.model = deepforest.deepforest()
        self.model.use_release()

    def predict_with_saved_model(self, model_name, image_name):
        self.model = deepforest.deepforest(saved_model=model_name)
        img = self.model.predict_image(image_name, return_plot=True)

        pyplot.imshow(img[:, :, ::-1])
        pyplot.show()

    def predict_and_store_trees_from_image(self,
                                           image_name,
                                           folder_name=None,
                                           score_threshold=0,
                                           tree_count_threshold=None,
                                           start_count_from=None):

        if folder_name is None:
            folder_name = os.path.join(paths.OUTPUT_DIR, image_name.split(sep=".")[0])

            # Create folder to save result
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print("Created folder:", folder_name)

        image_name = os.path.join(paths.DATA_DIR, image_name)

        print("Save folder:", folder_name)
        print("Predicting trees...")

        bounding_boxes = self.model.predict_image(image_name, return_plot=False)

        tree_count = len(bounding_boxes)

        if tree_count_threshold is not None:
            if tree_count < tree_count_threshold:
                print("Trees found:", tree_count)
                print("Tree count is lower than the threshold")
                return
        elif tree_count == 0:
            print("No trees found")
            return 0

        data = {"trees": []}
        tree_img = Image.open(image_name)
        picture_count = 0
        saved_images = 0

        if start_count_from is not None:
            picture_count = start_count_from

        for index, row in bounding_boxes.iterrows():

            img_name = "tree_" + str(picture_count)
            x_min = round(row["xmin"])
            y_min = round(row["ymin"])
            x_max = round(row["xmax"])
            y_max = round(row["ymax"])
            score = round(row["score"], 2)

            if score < score_threshold:
                continue

            data["trees"].append({
                "tree_name": img_name,
                "x_min": + x_min,
                "x_max": + x_max,
                "y_min": + y_min,
                "y_max": + y_max,
                "score": + score
            })

            cropped_img = tree_img.crop((x_min, y_min, x_max, y_max))
            cropped_img.save(os.path.join(folder_name, img_name + ".png"))
            print("Saved:", os.path.join(folder_name, img_name + ".png"))
            saved_images += 1
            picture_count += 1

        json_file = open(os.path.join(folder_name, "json.txt"), "w")
        json.dump(data, json_file)
        print("Pictures saved:", saved_images)
        return saved_images

    def predict_and_store_trees_from_folder(self, folder_name, save_folder):

        save_folder = os.path.join(paths.OUTPUT_DIR, save_folder)

        # Create folder to save result
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        picture_count = 0

        folder_path = os.path.join(paths.DATA_DIR, folder_name)
        images = len(os.listdir(folder_path))

        for index, image in enumerate(os.listdir(folder_path), start=1):
            print("Image ", index, " of ", images)
            picture_path = os.path.join(folder_path, image)
            picture_count += self.predict_and_store_trees_from_image(picture_path,
                                                                     folder_name=save_folder,
                                                                     score_threshold=0.55,
                                                                     start_count_from=picture_count)

    def evaluate_model(self, csv_file, model=None, weights=None):

        annotations_file = get_data(csv_file)
        mAP = model.evaluate_generator(annotations=annotations_file)
        print("Mean Average Precision is: {:.3f}".format(mAP))


predictor = Predictor()

#predictor.predict_and_store_trees_from_folder("trees", "test")
#predictor.predict_and_store_trees_from_image("DJI_0003.JPG", score_threshold=0.4)
