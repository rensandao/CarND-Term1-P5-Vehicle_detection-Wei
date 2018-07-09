## Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/car.png
[image4]: ./output_images/RGB_histogram.png
[image5]: ./output_images/scale1.png
[image6]: ./output_images/scale2.png
[image7]: ./output_images/scale3.png
[image8]: ./output_images/win_heat_final_1.png
[image12]: ./output_images/win_heat_final_2.png
[image13]: ./output_images/win_heat_final_3.png
[image14]: ./output_images/win_heat_final_4.png
[image15]: ./output_images/win_heat_final_5.png
[image16]: ./output_images/win_heat_final_6.png

[image9]: ./output_images/find_car.png
[image10]: ./output_images/final_img.png
[image11]: ./output_images/img_in_video.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first step "1.Features Extraction" of the IPython notebook. 

I started by reading in all the `vehicle` and `non-vehicle` images. The numbers of them were` 8792 vehicles` and `8968 non-vehicles` in respective. Basically, the dataset keep a balance. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then I took histogram in R,G,B channel respectively to see their features. Every single channel had different distribution which can be used to classify lately. Here are an example map:

![alt text][image3]

![alt text][image4]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. 

Here is an example with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Then I defined those functions for features extraction. Among them, the function `color_hist` computes color histogram features. The function `bin_spatial` computes binned color features. The function `get_hog_features` returns HOG features and visualization. And `extract_features` combined all the functions above to augment data features.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters in color spaces and HOG paramenters for extracting features and SVM classifier. YUV and YCrCb showed good effect, with a little variation. Considering the accuracy and real effect, I finally chose `YCrCb` color space. And orientation = 9, pix_per_cell = 8 and cell_per_block = (2,2). 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG and color features. The code can be found in second step "2.Classifier Traning". The parameters used `9 orientations 8 pixels per cell and 2 cells per block`. Feature vector length became `6108`. And the test accuracy of SVC came to `98.73%`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled ‘Find cars in images’ in third step , I defined a function `find_cars`, searching in a part of image in account of irrelative place such as sky and trees above the image. The searching district was decided by setting `ystart` and `ystop`. This can greatly reduce the amount of calculation and improve the accuray of detection.

considering the varying far distance to other cars, the further a car,the smaller the window. I set three kind size of scales to search a car. Here are three examples under size of 1.0, 1.5, 2.0 in digital. 

scale = 1.0

![alt text][image5]

scale = 1.5

![alt text][image6]

scale = 2.0

![alt text][image7]


Instead of using overlap, windows slided in term of cells per step, which is set as 2. Here is an example for testing `find_cars`: 

![alt text][image9]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap using function
`add_heat` and then thresholded that map to identify vehicle positions using `apply_threshold`.  I then used `label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `label(heatmap)` and the bounding boxes then overlaid on the last frame of video:

The corresponding heatmaps can be seen above.

### Here the resulting bounding boxes in six test images:
![alt text][image10]

### Here is the real effect map in the video:
![alt text][image11]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

For Features Extraction and classifier training, to get different features for classifying, I applied three ways including spatial bins, color histograms and hog features. But parameter selection seems to be many trial and hard work. And the result cannot certainly becomes better as it can be suit for all cases. This is an uncertain place for me and I think I need to try more experiment and achieve some experience. 

In the section of slide window, choosing different scales size and search aeras also spent me a lot of time. And the result seems to be 
uncertain. I guess I haven't set good parameter values and strategies.

In [my video result](./project_video_out.mp4), `Pipeline` seems to fail recognizing and drawing boxes in some clip. This made me uneasy, as it 
indicates great danger if it were in real situation. So more detailed work need to be done to improve the result.

Next, I will try some other classfier or combined method to improve the accuracy and robustness, and have some adjustment in searching aeras strategies.


### Reference
[Udacity]https://github.com/udacity/CarND-Vehicle-Detection.git

### change log



