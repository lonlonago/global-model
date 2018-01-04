# -*- coding: utf-8 -*-


def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。·
    """
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2]-Reframe[0] + 1
    height1 = Reframe[3]-Reframe[1] + 1

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2]-GTframe[0] + 1
    height2 = GTframe[3]-GTframe[1] + 1

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height; # 两矩形相交面积
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);

    # return IOU
    return ratio



def IOU_V2(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    简化了计算过程
    """
    width1 = Reframe[2]-Reframe[0] + 1
    height1 = Reframe[3]-Reframe[1] + 1

    width2 = GTframe[2]-GTframe[0] + 1
    height2 = GTframe[3]-GTframe[1] + 1

    iw = (
        min(Reframe[2], GTframe[2]) -
        max(Reframe[0], GTframe[0]) + 1
    )
    ih = (
        min(Reframe[3], GTframe[3]) -
        max(Reframe[1], GTframe[1]) + 1
    )

    if iw <=0 or ih <= 0:
        ratio = 0 # 重叠率为 0
    else:
        # 两矩形相交面积 / 总面积
        ratio = iw * ih / (width1*height1 + width2*height2 - iw * ih )

    # return IOU
    return ratio




def IOU_x1y1wh_xyxy(Reframe,GTframe):
    """
    自定义类似 matlab 代码的函数，计算两矩形 IOU，传入分别为x1y1wh和xyxy 格式数据。·
    xywh 和 x1y1wh二者有区别，并不是传入的中心位置，而是左上角和长宽。。。
    """
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2]  #-Reframe[0];
    height1 = Reframe[3] #-Reframe[1];

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2]-GTframe[0];
    height2 = GTframe[3]-GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0
    else:
        Area = width*height; # 两矩形相交面积
        Area1 = width1*height1;
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    # return IOU
    return ratio
