# Drought Watch
## Efficientnet-based convolutional neural network prediction of foliage quality from sattelite images
### Please see video for high-level overview and this readme for technical details and in-depth explanations
### Deployed at [https://www.stanleyzheng.ca](https://www.stanleyzheng.ca)

## Inspiration
Droughts cause an extraordinary amount of damage yearly - we were inspired by a [paper presented at ICLR](https://arxiv.org/abs/2004.04081) titled "Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya", which I heard of from Kaggle. 
A corresponding competition was launched on [Weights and Biases](https://wandb.ai/wandb/droughtwatch/benchmark). Our hope is that this model can lessen the impact of climate change by providing aide organizations a way to gauge the severity of drought, and therefore, allocate resources more effectively. The ease of use of this application means that this model could also be used directly by shepherds to find the best grazing area in a large area.

## What it does
Our model takes 10 channels of light and predicts foliage levels. Each image is 65x65 and represents about a 2km square, and is classified into 4 foliage levels: 0 means that no cows can graze there, and 3 means that 3 or more cows can graze there. Through this, pastoralists can better-direct their pasture, and since foliage quality is a determining factor in drought prediction, large surveys can be made with this model in order to predict severity of drought over a large area. Furthermore, data and predictions can be used to aid research projects.

## How I built it
The model itself was built in Tensorflow. I usually use PyTorch, but given the ease of development of Tensorflow and the fast deployment, it was superior for this hackathon. The model is an efficient-b0 based model with channels stretched to 10 trained on a Nvidia Tesla V100 for about 40min. We initially considered using TPU's for lower training time, but development time was too large and GPU's were adequately fast. ResNeSt was also tried, though it was slow to run on our Heroku web-hosting, which has only 500mb of memory and a single core CPU. The website was done completely in Python as well, using a library called Streamlit. TTS was done automatically by Weights and Biases, and we used tfrecords premade for their competition. The dataset was quite amazing, with great quality - already split into tfrecords which could be rapidly loaded, and with over 120 000 samples. 

## Accomplishments that I'm proud of
I'm quite proud of making an end-to-end machine learning web app in just 12 hours. I applied many skills learned from Kaggle, and a lot of boilerplate code in order to achieve this. While our model doesn't achieve the State of the Art on this dataset, it handily defeats currently-deployed models, and is only about 3% worse than the State of the Art on this task, a modified Efficientnet B3 model.

## What's next for Drought Watch
We'd like to do more data visualization - currently, our site only displays images for JPG's, and not 10 channel raw. This was due to a 5x latency increase, which we didn't have time to optimize. In addition, our model could be improved in many ways - bayesian optimization with more robust validation, possibly kfold, would make the model more robust and could optimize further. We didn't have much time to experiment much, but perhaps a larger model could be deployed as well. Finally, support for multiple images inside of one .npy file would be desirable.