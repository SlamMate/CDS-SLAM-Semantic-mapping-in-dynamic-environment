# CDS-SLAM-Semantic-mapping-in-dynamic-environment
## 1. Introdution
This project is the result of my undergraduate dissertation. The localization in a dynamic environment is to deploy TensorRT optimized YOLOX in the front end of ORB-SLAM3 for object detection and eliminate all points belonging to the human bounding box. At the same time, the semantic information is sent to the mapping module to dye the 3D point cloud. The disadvantage of this project is that in the localization module, only the points belonging to people are processed because people are dynamic most of the time. In the mapping module, we did not segment semantic objects accurately, resulting in the wrong coloring of point clouds of other objects.

The main part of the code will be published after the paper based on this is received by publication.

The video is presented below:

China:https://www.bilibili.com/video/BV1St4y157qH?share_source=copy_web&vd_source=12d45e19826de0471391d3db9d6c9491

Other countries:https://www.youtube.com/watch?v=OxYHrIgqyJQ

## 2. Acknowledgements

My work is based on ORBSLAM3,Crowd-SLAM,Semantic-SLAM
  
    @article{ORBSLAM3_TRO,
      title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial 
               and Multi-Map {SLAM}},
      author={Campos, Carlos AND Elvira, Richard AND G\´omez, Juan J. AND Montiel, 
              Jos\'e M. M. AND Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics}, 
      volume={37},
      number={6},
      pages={1874-1890},
      year={2021}
     }
     @article{soaresJINT2021,
      title={Crowd-{SLAM}: Visual {SLAM} Towards Crowded Environments using Object Detection},
      author={Soares, J. C. V., Gattass, M. and Meggiolaro, M. A.},
      journal={Journal of Intelligent & Robotic Systems},
      volume={102},
      number={50},
      doi = {https://doi.org/10.1007/s10846-021-01414-1},
      year={2021}
     }
     @inbook{inbook,
     author = {Qi, Xuxiang and Yang, Shaowu and Yan, Yuejin},
     year = {2018},
     month = {10},
     pages = {012023},
     title = {Deep Learning Based Semantic Labelling of 3D Point Cloud in Visual SLAM},
     volume = {428},
     journal = {IOP Conference Series: Materials Science and Engineering},
     doi = {10.1088/1757-899X/428/1/012023}
     }
     
  if you use this to research you must cite this paper. The paper is based on this code
  
  @article{10.1088/1361-6501/acd1a4,
	author={Zhang, Qi and Li, Changdi},
	title={Semantic SLAM for mobile Robots in dynamic environments Based on visual camera sensors},
	journal={Measurement Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/1361-6501/acd1a4},
	year={2023},
	abstract={Visual SLAM is constrained by a strong static world assumption,&#xD;making it hardly succeed when dynamic objects appear. Although some methods&#xD;eliminate dynamic objects by combining semantic and geometric information, the&#xD;fewer classes detected by the semantic method can not benefit these systems by&#xD;applying more scenes and constructing a richer semantic map. Our paper raises a&#xD;semantic visual SLAM system for a motion scene, which can run in real-time. We&#xD;use object detection to detect 80 classes in the scene and the moving consistency&#xD;checking to find outliers in every image. We propose different methods to check&#xD;the motion state of humans and other objects separately. For the detected human,&#xD;we propose an algorithm that judges if a person is sitting and divides the bounding&#xD;box belonging to the sitting person into two parts according to the proportion of&#xD;the human body. Then we use the same threshold as checking the boxes of the&#xD;standing person to determine the state of the two boxes belonging to the sitting&#xD;person, respectively. For other different objects except for humans, we propose an&#xD;algorithm that automatically adjusts the threshold of different bounding boxes.&#xD;This way, it can have the same detection performance for different objects. Finally,&#xD;we retain the static box points contained in the dynamic box and eliminate other&#xD;points in the dynamic box to benefit system performance from more detected&#xD;classes. We evaluated our SLAM on the TUM RGB-D dataset. The performance&#xD;of our SLAM exceeds most SLAM systems in a dynamic environment. Our system&#xD;is also tested in a real environment with a monocular camera to show its robustness&#xD;and universality.}
}
  ## 3.Prerequisites
  Please refer to the compilation and operation of ORB-SLAM3:https://github.com/UZ-SLAMLab/ORB_SLAM3

