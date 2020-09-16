# Hand-Pose-Prediction
This repository is for research purpose and to bring optimizations in the 3d-hand pose estimations techniques that have already been implemnted by 
Researchers.

STRATEGY.

Depth Images from kinect model will be obtained.

Depth image will be transformed into volumetric representation at a very low latency by cuda programming and using multiple cuda cores in GPU.

Neural Network with 3D convolutions will be used to exploit and perfrom regression on the Hand joints.

Various Supervied and Unsupervied Methods will be used and optimizations will be made in the precedding research:


Refrences:

Real-time 3D Hand Pose Estimation
with 3D Convolutional Neural Networks
Liuhao Ge, Hui Liang, Member, IEEE, Junsong Yuan, Senior Member, IEEE, and Daniel Thalmann

A. Haque, B. Peng, Z. Luo, A. Alahi, S. Yeung, and F.-F. Li, “Towards
viewpoint invariant 3D human pose estimation,” in Proc. Eur. Conf.
Comput. Vis., 2016, pp. 160–177

The strategies in these papers are to be implemented  .

ABSTRACT:

With the increase in the applications of VR/AR, 3D hand gestures are increasingly being used as an actuation input.
Consequently, hand pose estimation is an active field of interest today. 
Hand pose estimation using RGB camera(s) has immense potential due to its ready availability and low price point.
This group aims to contribute to optimization of existing 3D hand pose estimation models that use RGB cameras.
Typical RGB camera-based systems have higher latencies and lower accuracy compared to systems with specialized cameras. 
This group aims to develop algorithms/techniques to minimize their latencies and improve their accuracy. 
This would offer a low-cost visual computing-based counterpart to existing systems.
Furthermore, a computer vision-based technique would be more natural and affordable compared interfaces that use gloves which are expensive, large and heavily wired.
As a final outcome, this group will like to demonstrate the efficacy of this product by showcasing an actuation use-case






