
Real Mask Dection - v18 2022-11-22 12:53am
==============================

This dataset was exported via roboflow.com on November 21, 2022 at 5:57 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 6987 images.
Masked are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Fit (black edges))
* Grayscale (CRT phosphor)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -5 and +5 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Salt and pepper noise was applied to 1 percent of pixels


