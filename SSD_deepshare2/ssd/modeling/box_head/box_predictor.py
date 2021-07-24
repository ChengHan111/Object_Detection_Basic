import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry


class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        # 6 个feature maps, 为不同feature maps做预测for循环得到的结果
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous()) # 我们在这里拿到六个feature maps都做完卷积之后的值 如第一个feature map的值为[16, 38, 38, 84] 16 是batch_size
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous()) # 同上, 可以在下面一行设置断点进行查看

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    # 分类的卷积操作
    def cls_block(self, level, out_channels, boxes_per_location):
        # 38 x 38 x(21 x 4) for the first feature map since we have 4 anchor boxes for the first feature map. Remember [4 6 6 6 4 4]
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    # 回归的卷积操作
    def reg_block(self, level, out_channels, boxes_per_location):
        # also take the first feature map in to consideration 4 x 4 = 16 therefore the first feature map get the output as 38x38x16
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
