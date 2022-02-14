# Using-Eye-Gaze-to-Forecast-Human-Pose-in-Everyday-Pick-and-Place-Actions

We present a novel method that uses gaze information to predict the object being fixated, as well as the human pose the instant the fixated object is picked or placed. Published at the International Conference on Robotics and Automation (ICRA) 2022.

Also check out the method that does not use eye-gaze but instead, predicts the human pose given the coordinates of every object in the scene. Published at the International Conference on Acoustics, Speech, & Signal Processing (ICASSP) 2022. [[Code]](https://github.com/HaziqRazali/Using-a-Single-Input-to-Forecast-Human-Action-Keystates-In-Everyday-Pick-and-Place-Actions)

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Results](#results)
  * [Visualization](#visualization)
  * [Training](#training)
  * [Testing](#testing)
  * [Future Work](#usage)
  * [References](#references)

# Requirements
------------

What we used to develop the system

  * ubuntu 18.04
  * anaconda
  
For visualization

  * pyqtgraph
  * pyopengl
  
For training and testing

  * pytorch
  * tensorflow
  * tensorboardx
  * opencv

# Brief Project Structure
------------

    ├── data          : directory containing the data and the scripts to visualize the output
    ├── dataloader    : directory containing the dataloader script
    ├── misc          : directory containing the argparser, loss computations, and checkpointing scripts
    ├── model         : directory containing the model architecture
    ├── results       : directory containing some sample results
    ├── shell scripts : directory containing the shell scripts that run the train and test scripts
    ├── weights       : directory containing the model weights
    ├── test.py       : test script
    ├── train.py      : train script
    
# Results
------------

  * Given the input pose in white, the gaze vectors in purple, and the coordinates of all objects in the scene, our method first attends to fixations and the fixated object, before predicting the pose the instant the person picks or places the fixated object.

<img src="/misc/1.png" alt="1" width="400"/> <img src="/misc/2.png" alt="1" width="400"/>

# Visualization
------------

  * To visualize a sample output shown above, download the [processed data](https://imperialcollegelondon.box.com/s/71ki6qw8hjr3olyzmy05lye6le1bqyk5) then unzip it to `./data` as shown in [Brief Project Structure](#brief-project-structure) and run the following commands:
 
```
conda create -n visualizer python=3.8 anaconda
conda install -c anaconda pyqtgraph
conda install -c anaconda pyopengl
cd data/scripts/draw-functions
```

  * The following command visualizes the input pose, gaze-vectors as well as all objects in the scene for the pick action (Left Figure).

```
python draw-pred-eye-gaze.py --frame 2600 --draw_true_pose 0 --draw_pred_pose 0 --object_attention 0 --gaze_attention 0
```

  * The following command visualizes the attended gaze-vectors and objects, as well as the predicted pose for the pick action (Right Figure).

```
python draw-pred-eye-gaze.py --frame 2600 --draw_true_pose 0 --draw_pred_pose 1 --object_attention 1 --gaze_attention 1
```

  * The following commands visualizes the results for the place action.

```
python draw-pred-eye-gaze.py --frame 2180 --draw_true_pose 0 --draw_pred_pose 0 --draw_grid 0 --gaze_attention 0
python draw-pred-eye-gaze.py --frame 2180 --draw_true_pose 0 --draw_pred_pose 1 --draw_grid 1 --gaze_attention 1
```

  * Note that the `./results` folder already contain some sample results.

# Training
------------

  * To train a model, download the [processed data](https://imperialcollegelondon.box.com/s/71ki6qw8hjr3olyzmy05lye6le1bqyk5) then unzip it to `./data` as shown in [Brief Project Structure](#brief-project-structure) and run the following commands:

```
conda create -n forecasting python=3.8 anaconda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tensorflow
conda install -c conda-forge tensorboardx
conda install -c conda-forge opencv

cd shell_scripts/ICRA2022
./eye-gaze-train.sh
```

  * The weights will then be stored in `./weights`

# Testing
------------

  * To test the model, run the following command:

```
./eye-gaze-test.sh
```

  * The outputs will then be stored in `./results` that can be visualized by following the commands listed in [Visualization](#visualization). Note that the `./weights` folder already contain a set of pretrained weights.

# References
------------

```  
@InProceedings{haziq2022eyegaze,  
author = {Razali, Haziq and Demiris, Yiannis},  
title = {Using Eye Gaze to Forecast Human Pose in Everyday Pick and Place Actions},  
booktitle = {International Conference on Robotics and Automation},  
year = {2022}  
}  
```
