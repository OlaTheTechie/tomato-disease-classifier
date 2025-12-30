# tomato-disease-classifier
A system for detecting tomato plant diseases using deep learning


## Problem

Tomato plants are highly susceptible to various disease such as late blight, early blight, septoria leaf spot, bacteria leaf spot, and tomato mosaic virus.
These diseases reduce crop quality and yield causing severe economic losses for farmers, especially smallholder farmers who lacks access to expert agronomic support.

Traditional disease diagnosis relies on manual inspection by experts which pressents various problems: 

1. Requires agricultural expertise which is always unavailable or expensive
2. Time-consuming and prone to human error
3. Difficult to scale across large farms or remote regions
4. Delayed diagnosis leads to disease spread and irreversible crop damage

There is a critical need for an automated, accurate, and scalable system that can detect tomato diseases early using images of plant leaves.



## Solution 

This project proposes a deep learning based image classification system using convolutional neural networks (CNNs) to automatically detect and classify tomato leaf diseases from images. 

## System Overview
1. Data collection 
    - Use already labelled tomato leaf images covering: 
        - Healthy leaves
        - Multiple disease classes (e.g., early blight, late blight, leaf mold, etc.)
    
    - Public datasets such as PlantVillage or custom field images can be used. 

2. Data Processing 
    - Image resizing and normalisation
    - Data augmentation (rotation, flipping, brightness adjustment) to improve generalisation
    - train - validation -test split

3. Model architecture 
    - CNN based architecture such as: 
        - Custom CNN 
        - Transfer learning (e.g., ResNet, MobileNet, EfficientNet)
    - Convolutional layers for feature extraction
    - Fully connected layers for classification


4. Training and optimisation
    - Loss function: categorical cross-entropy
    - Optimizer: adam / SGD
    - Regularisation: Dropout, Batch normalisation
    - Early stopping to prevent overfitting

5. Inference and Depployment
    - Predict disease class from a single leaf image
    - Potential deployment as: 
        - Web application 
        - Mobile application 
        - Edge device soltion for farms


