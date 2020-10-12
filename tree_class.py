import os
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math
import reverse_geocode

class tree_data:
    def __init__(self, data):
        self.data = data
        self.dest = './image_labeled_input'
        
    def show_labeled(self,label):
        dirpath = self.dest + '/' + label
        path = dirpath + '/*.png'
        #print(path)
        #print(data)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            if not os.listdir(dirpath):
                print("Directory is empty")
                return
            else:    
                pass
                #print("Directory is not empty")
        else:
            print("Given Directory doesn't exist")
            return

        #images = [cv2.imread(file) for file in glob.glob(path)]
        image_filenames = [os.path.basename(file) for file in glob.glob(path)]
        #print(image_filenames)
        #images = [cv2.imread(file) for file in glob.glob(path)]
        image_filenames.sort()
        
        imgs = []
        for image in image_filenames:
            imgs.append(cv2.imread(self.dest + '/' + label + '/' + image))

        if len(image_filenames) < 1:
            print("No images in folder")
            return

        length = math.sqrt(len(image_filenames))
        gridsize = math.ceil(length)

        # make grid for all tree images. Figure size scale with gridsize
        # The grid will always have extra unnecessary plots
        fig, axs = plt.subplots(gridsize, gridsize, figsize=(gridsize*2,gridsize*2))
        im = 0
        for i in range(0, gridsize):
            for j in range(0, gridsize):

                # delete unnecessary plots
                if im >= len(image_filenames):
                    fig.delaxes(axs[i][j])
                else:
                    axs[i,j].imshow(imgs[im])
                    axs[i,j].set_title(image_filenames[im])
                    axs[i][j].set_xticks([])
                    axs[i][j].set_yticks([])

                    im = im + 1
    def show_forest_cluster(self):
        
        data = self.data
        
        forest_img = './forests/' + data['image']

        parent_image = cv2.imread(forest_img)

        trees = data['trees']

        color = (255,0,0)

        # draw rectangles
        for tree in trees:
            parent_image = cv2.rectangle(parent_image, (tree['x_min'], tree['y_min']),(tree['x_max'], tree['y_max']), color, 5)


        plt.figure(figsize = (18,12))
        plt.axis('off')
        plt.imshow(parent_image, aspect='auto')
        plt.show()
    
    def set_label(self,species):
        if(species == "skip"):
            skipLabel()
            return

        trees = self.data['trees']
        forest_image = self.data['image']
    
        path = self.dest + '/' + species
        try:
            os.mkdir(path)
        except OSError:
            pass
            #print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

        # remove file ending
        if forest_image.endswith('.JPG'):
            forest_image = forest_image[:-4]

        # copy tree to labels
        for tree in trees:
            from_path = './images_and_json/' + forest_image + '/' + tree['tree_name'] + '.png'
            to_path = path + '/' + tree['tree_name'] + '.png'
            #print(from_path, to_path)
            shutil.copy(from_path, to_path)


        print('Moved trees to label ' + species)
        
    def delete_labeled(self, label, img_name):
        path = self.dest + '/' + label + '/' + img_name
        print(path)

        if not os.path.exists(path):
            print("Image doesn't exist")
            return
        try:
            os.remove(path)
            print("Succesfully deleted " + img_name)
        except:
            print("Error, failed to delete " + img_name)
            
    def dms_to_dd(self, d, m, s):
        dd = d + float(m)/60 + float(s)/3600
        return dd
    
    
    def show_location(self):
        lat = self.data['latitude']
        long = self.data['longitude']
        
        try:
            latt = self.dms_to_dd(lat[0],lat[1],lat[2])
            longg = self.dms_to_dd(long[0],long[1],long[2])

            #print(latt, longg)
            coordinates = (latt, longg)
            location = reverse_geocode.get(coordinates)
        except:
            print("Can't determine location")
            return
        else:
            print("This forest is near " + location['city'] + " in " + location['country'])