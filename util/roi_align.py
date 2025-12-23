import numpy as np
import torch
from torch import nn
import torchvision


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height  # 高
        self.crop_width = crop_width  # 宽
        self.extrapolation_value = extrapolation_value  # 外推值
        self.transform_fpcoor = transform_fpcoor  #

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """

        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)  # torch.split()作用将tensor分成块结构。

        image_height, image_width = featuremap.size()[2:4]
        if self.transform_fpcoor:

            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)
            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)  # 翻转图片？

        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)

        # boxes = boxes.detach()
        # print( id(boxes) )
        # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        # contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据
        boxes = boxes.detach().contiguous()
        box_ind = torch.unsqueeze(box_ind, 1)
        box_ind = box_ind.detach()
        boxes = torch.cat((box_ind, boxes), 1)

        # return torchvision.ops.roi_align(feature_map, roi_data, [self.crop_height, self.crop_width], spatial_scale=1.0,
        #                                  sampling_ratio=-1)
        return torchvision.ops.roi_align(featuremap, boxes, [self.crop_height, self.crop_width], spatial_scale=1.0,
                                         sampling_ratio=-1)
        # return CropAndResizeFunction.apply(featuremap, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)
#
# feature_map = np.arange(64).reshape(8, 8)
# roi_data = (0.6, 1.6, 9.2, 11.0)
#
# print(torchvision.ops.roi_align(feature_map, roi_data, [2, 2], spatial_scale=1.0,
#                                          sampling_ratio=-1))
# roi_data = (0.6, 1.6, 9.2, 11.0)
# feature_map = np.arange(64).reshape(8, 8)
# boxes_features = roi_align(feature_map, roi_data, 1)

