# CDS-SLAM-Semantic-mapping-in-dynamic-environment
## Introdution
This project is the result of my undergraduate dissertation. The localization in dynamic environment is to deploy TensorRT optimized YOLOX in the front end of ORB-SLAM3 for object detection, and then eliminate all points belonging to the human bounding box. At the same time, the semantic information is sent to the mapping module to dye the 3D point cloud. The disadvantage of this project is that in the localization  module, only the points belonging to people are processed, because people are dynamic most of the time. In the mapping module, we did not segment semantic objects accurately, resulting in wrong coloring of point clouds of other objects.

The main part of code will be published after the paper based on this is received

The video is presented below:

https://www.bilibili.com/video/BV1St4y157qH?share_source=copy_web&vd_source=12d45e19826de0471391d3db9d6c9491
