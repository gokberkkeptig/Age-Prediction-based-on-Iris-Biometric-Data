# Age-Prediction-based-on-Iris-Biometric-Data

# Objectives: 
The purpose of this project is to familiarize with the fundamental deep 
learning solutions to computer vision problems and framework on age prediction problem. The 
assignment aims to give insights about the deep learning based computer vision research and their 
evaluation methods.
# Description: 
In this project I implement an age prediction system based on deep
learning methods, and to evaluate it with the provided dataset.
The text continues with detailed explanations of the methods and requirements.
# 1. Age prediction based on iris biometric data
The main purpose of the age prediction systems is to determine age group (group1: <25, group2: 25-
60 and group3: >60 ) of the person in a query image. The prediction is done by evaluating semantic 
contents of the query image. However, there is a diffculty in revealing the semantics of images due 
to the semantic gap. In order to overcome this diffculty, images are described as feature vectors 
which are higher level representations than collection of numbers. 
With these feature vectors, age prediction can be formulated as a learning problem to match an 
image representation with the age group of person in the image. Hence, in this assignment you are 
required to construct a fully-connected network with rectified linear unit (ReLU) as nonlinearity 
function between layers and train it with RMSprop optimizer using the provided feature vectors.
While training the network I use softmax(cross-entropy loss) function to minimize 
the difference between actual age group and the estimated one.
# 2. Dataset and feature extraction
The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilised 
for this project. Four eye images (two left and two right) were acquired in two different sessions with 
a resolution of 640*480 pixels. The 200 subjects providing the samples contained in this database are 
within the age range of 18-73.The training and the testing sets were formed to be person-disjoint sets. Approximately 72% of the 
subjects in each age group are used for training and the remaining subjects used as a testing set. The 
available number of subjects in the testing and the training sets for each age group is shown in the 
following Table.

Sets Age groups
<25 25-60 >60
All 70 115 15
Training 50 82 11
Testing 20 33 4

For this project three different types of iris biometric features will be used as in my previous work
• Texture features: These are features which describe the pattern of the iris available only 
from the overall finished output of the acquisition, segmentation, normalisation and feature 
extraction process respectively. 
• Geometric features: These are features which describe the shape (physical appearance) of 
the iris, and are thus available only from the output of the acquisition and segmentation 
process respectively.
• Both geometric and texture features: simply is the combination (concatanation) of both 
feature types.
