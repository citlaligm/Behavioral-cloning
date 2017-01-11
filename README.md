### Behavioral Cloning

#Overview
This project aims to develop and train a deep learning network that learns how to drive a car around a track. The training data was provided by the Udacity team. When the model was ready I tested it running the simulator in autonomus mode, the car succesfully drove around the track.

The main parts of the code:
1. Preprocessing trainig data.
3. Normalization of the data.
4. Augmentation methods.
5. Define NVIDIA model.

I'd like to thank this paper from NVIDIA [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and [this blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28) post for the great help they provide.


#Preprocessing data training
The training data originally had a size of 160x320x3, since each image has information that is not relevant for driving, like the sky or the trees. I cropped the top 1/5 of the image and in the bottom I cropped 25 pixels to remove the hood of the car.

The following image is an example of the results.

![Original Image](images/original.png)

![Crop Image](images/crop.png)


## Deployment

Add additional notes about how to run this

## Authors



## License


## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

