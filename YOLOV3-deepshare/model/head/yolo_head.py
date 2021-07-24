import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC #20
        self.__stride = stride


    def forward(self, p): # 4x75x52x52 对于52x52输入来说 batch_size=4 75 = 3*（20+5）
        bs, nG = p.shape[0], p.shape[-1]
        # [4(batch_size), 3, （5+20）, scale, scale] ---> [4(batch_size),scale,scale,3,（5+20）]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)

        p_de = self.__decode(p.clone())

        return (p, p_de)


    def __decode(self, p):
        # get batch_size and output_size from above
        batch_size, output_size = p.shape[:2]

        device = p.device
        # 8/16/32
        stride = self.__stride
        # 这里我们的anchors参数已经进行了变换，如果没有进行变化这里要乘以系数
        anchors = (1.0 * self.__anchors).to(device)

        # x,y
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        # h,w
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        # objectness
        conv_raw_conf = p[:, :, :, :, 4:5]
        # 6-25 20for VOC classes
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1) # 生成坐标格
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device) #repeat 3 次因为要对三个不同的scale分别生成一张网

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride #grid_xy 锚框中心点
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride #anchors锚框的初始化宽高
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf) # [0, 1]范围内
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1) # 拼接完仍然是25维结果

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
