from numpy import average, zeros
from functions import *
import pandas as pd

cascades_list, cascades_names=getCascadesAndNames()
test_images, test_images_names=getTestImagesAndNames()

cascades_data=pd.DataFrame(cascades_names)
cascades_data.insert(1, "actual_num",0)
cascades_data.insert(1, "detected_num",0)
cascades_data.insert(1, "average_accuracy",0.0)
cascades_data.insert(1, "hard_accuracy",0.0)
cascades_data["detected_num"] = pd.to_numeric(cascades_data["detected_num"])
cascades_data["actual_num"] = pd.to_numeric(cascades_data["actual_num"])
cascades_data["average_accuracy"] = pd.to_numeric(cascades_data["average_accuracy"])
cascades_data["hard_accuracy"] = pd.to_numeric(cascades_data["hard_accuracy"])

print(cascades_data)

for (frame, image_name) in zip(test_images, test_images_names):
    print("Detecting on image: ", image_name)
    row=0
    for (cascade_filter, cascade_name) in zip(cascades_list, cascades_names):
        
        detected_objects=detectObjects(frame,cascade_filter, 24)
        detected_number_of_cars=len(detected_objects)
        actual_number_of_cars=getNumOfCarsOnImage(image_name, "Annotations\pos.txt")
        current_accuracy=calculateAccuracy(detected_number_of_cars, actual_number_of_cars)
        
        print("Cascade '",cascade_name,"' found ", len(detected_objects), "out of ", actual_number_of_cars, " cars.")
        
        cascades_data["detected_num"][row]+=detected_number_of_cars
        cascades_data["actual_num"][row]+=actual_number_of_cars
        cascades_data["hard_accuracy"][row]=calculateAccuracy(detected_number_of_cars,actual_number_of_cars)
        if(cascades_data["average_accuracy"][row]==0):
            cascades_data["average_accuracy"][row]=current_accuracy
        else:
            cascades_data["average_accuracy"][row]=(current_accuracy+cascades_data["average_accuracy"][row])/2
        
        row+=1    
    print(cascades_data)
    