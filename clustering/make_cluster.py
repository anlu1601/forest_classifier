import clustering.model as cl
import random as rand

def make_cluster(image):
    img_folder = r"./tree_segmentation/output/" + image
    pred, imgs = cl.run_INCEPTION_3(img_folder, "./clustering")
    #print(pred)


    first = rand.randrange(len(pred)-1)
    second = rand.randrange(len(pred)-1)
    print("First, second", first, second)
    simx, nnx = cl.get_nearest_neighbor_and_similarity(pred, len(pred), pred[first], "./clustering", 2)
    simy, nny = cl.get_nearest_neighbor_and_similarity(pred, len(pred), pred[second], "./clustering", 2)
    area, x, y = cl.get_areaFromEquation(simx, simy, nnx, 1, 0.3)

    image_cluster = []


    for a in area:
        print("IMAGE: ", imgs[a])
        image_cluster.append(imgs[a])

    return image_cluster