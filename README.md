# iterative_iou
Calculate iou for an arbitrary number of bounding box sets. Here "iou" is identical to the *dice* measurement of two set of pixels.
The algorithm is currently used for CIAC. And the ``README`` file will be modify in the near future for clarity.

**Note**: It is designed for [mmdetection3d](https://github.com/open-mmlab/mmdetection3d.git) since it depends on BaseInstance3DBoxes in ``mmdet3d.structures``. You can easily modify the code be compatible with your code.
