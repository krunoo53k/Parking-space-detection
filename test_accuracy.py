from functions import *
import copy

cascades_list, cascades_names=getCascadesAndNames()
test_images, test_images_names=getTestImagesAndNames()

for (frame, image_name) in zip(test_images, test_images_names):
    print("Detecting on image: ", image_name)
    for (cascade_filter, cascade_name) in zip(cascades_list, cascades_names):
        displayed_image=copy.deepcopy(frame)
        detected_objects=detectObjects(displayed_image,cascade_filter, 24)
        print("Cascade '",cascade_name,"' found ", len(detected_objects), "out of ", getNumOfCarsOnImage(image_name, "Annotations\pos.txt"), " cars.")