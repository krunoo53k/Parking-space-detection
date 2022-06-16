# Parking space detector

Detects how many cars are visible on an image of a parking lot using OpenCV and Viola-Jones algorithm.
The main goal of this project is to interface the  *"Drone-based Object Counting by Spatially Regularized Regional Proposal Networks"* [dataset](https://lafi.github.io/LPN/) with OpenCV through Python scripts.
This repo also contains several pre-trained haar-cascades and scripts for testing purposes.


# Folder structure
To properly use scripts in this repository you should abide by the following folder structure:
```
.
├── Annotations             			# Folder containing positional text file
├── cascades                			# Pre-trained cascades
├── datasets                			# Folder containing datasets
│   ├── CARPK_devkit        			
│       ├──  data
│            ├──  Annotations			# Containing annotation files for images
|			 ├──  Images		# Images for training/testing
├── negative  		 			# Folder which contains negative images for training process
├── test_images               			# Folder containing images from CARPK_devkit for testing purposes
├── LICENSE
└── README.md
```
If you choose to use a different folder structure, editing the scripts should be straightforward.
 

# How to run

Running each script and the process will be explained further.

## Create converted files

The *create_converted_files.py* should be ran first if there are no OpenCV compatible annotations. The following scripts loads the *negatives* folder, scanning them and creates a new *neg.txt* file containing the file-paths to each negative image. This is needed if one wants to train their model using *traincascades.exe* provided by the OpenCV library.
Furthermore, it reads the *Annotations* folder from the *CARPK* dataset and converts it to the one OpenCV expects. Images are annotated as *(x1, y1, x2, y2)* in the dataset, and OpenCV accepts the following format; *(x1, y1, width, height)*.

## Visual test

It is possible to visually test the accuracy of the model by using the *test_visual.py* script.
By default, it loads an image using a certain cascade, and displays the image on the screen, like this;
![Window showing detected cars](https://i.imgur.com/44sOkad.png)
However, this approach is slow, but it should give one a rough idea how a select model behaves.

## Test accuracy

It is also possible to test multiple models contained in the *cascades* folder on test images in the *test_images* folder. Just simply copy and paste images you wish to test from the dataset into the folder and run the *test_accuracy.py* script.
The script iterates through all of the models in the *cascade* folder while also using object detection on the test images. It will print out a table containing the names of cascades used, their accuracy, the number of detected objects and the total number of objects actually present on the image.
One distinction to make is that the *strict accuracy* column is referring to simply how much does the number of detected objects deviate from the number of total objects while the *average accuracy* column is just an average of all accuracy percentages on image-by-image basis.  



# References and thanks

Big thanks to [Meng-Ru Hsieh](http://www.cmlab.csie.ntu.edu.tw/~mru/), [Yen-Liang Lin](http://www.cmlab.csie.ntu.edu.tw/~yenliang/Home.html), [Winston Hsu](http://winstonhsu.info/) for providing the following [dataset](https://lafi.github.io/LPN/).
[Marinbenc](https://github.com/marinbenc) for helping me understand this assignment. 


