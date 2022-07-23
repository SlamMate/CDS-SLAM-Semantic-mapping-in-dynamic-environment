# CDS-SLAM-Semantic-mapping-in-dynamic-environment
## Introdution
This project is the result of my undergraduate dissertation. The localization in dynamic environment is to deploy TensorRT optimized YOLOX in the front end of ORB-SLAM3 for object detection, and then eliminate all points belonging to the human bounding box. At the same time, the semantic information is sent to the mapping module to dye the 3D point cloud. The disadvantage of this project is that in the localization  module, only the points belonging to people are processed, because people are dynamic most of the time. In the mapping module, we did not segment semantic objects accurately, resulting in wrong coloring of point clouds of other objects.

<iframe src="//player.bilibili.com/player.html?aid=983703014&bvid=BV1St4y157qH&cid=781449723&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
