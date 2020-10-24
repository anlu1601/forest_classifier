import os
import random as rand
import json


def get_trees_from_random_forest():
    forest_folders_dir = './tree_segmentation/output'

    dirlist = [item for item in os.listdir(forest_folders_dir) if os.path.isdir(os.path.join(forest_folders_dir, item))]
    # example: ('DJI_0001', 'DJI_0009', 'DJI_0123', 'DJI_0133', 'DJI_0250', 'DJI_0369', 'DJI_0465')

    ran = rand.randint(0, len(dirlist) - 1)
    # print(dirlist[ran])
    random_forest = dirlist[ran]
    # random_forest = 'DJI_0001'

    sub_images_json = './tree_segmentation/output/' + random_forest
    path_forest = './forests/' + random_forest + '.jpg'
    # print(path_forest)
    with open(sub_images_json + '/json.txt') as f:
        data = json.load(f)

    # print(data['image'])
    trees = data['trees']
    # print(data)
    return data

# print(trees)
# print(json.dumps(data, indent = 4, sort_keys=True))
# for tree in trees:
#     for dd in tree:
#         print(dd, tree[dd])
#     break

# for tree in trees:
#     print(tree['tree_name'])
#     break