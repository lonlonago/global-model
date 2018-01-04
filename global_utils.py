#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import PIL.ImageDraw as ImageDraw
import IOU



def prepare_global_label(img_path, xml_path):
    img = Image.open(img_path)

    if img.size[1] >= img.size[0]:
        large_side = img.size[1]
    else:
        large_side = img.size[0]

    scale_factor = 224.0 / large_side
    back_ground = Image.new('RGB', (large_side, large_side))

    bw, bh = back_ground.size
    ow, oh = img.size

    # pading image
    if img.size[1] >= img.size[0]:
        back_ground.paste(img, (int((bw - ow) / 2), int((bh - oh))))
        pad_size = int((bw - ow) / 2)
    else:
        back_ground.paste(img, (int(bw - ow), int((bh - oh) / 2)))
        pad_size = int((bh - oh) / 2)

    img_resize = back_ground


    # pading and resize the boxes
    boxes = []
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)  # - 1
        y1 = float(bbox.find('ymin').text)  # - 1
        x2 = float(bbox.find('xmax').text)  # - 1
        y2 = float(bbox.find('ymax').text)  # - 1
        # pading box
        if img.size[1] >= img.size[0]:
            x1 = x1 + pad_size
            x2 = x2 + pad_size
        else:
            y1 = y1 + pad_size
            y2 = y2 + pad_size
        # resize box
        x1 = x1 * scale_factor
        x2 = x2 * scale_factor
        y1 = y1 * scale_factor
        y2 = y2 * scale_factor

        boxes.append([x1, y1, x2, y2, 1])  # last is class

    boxes = np.array(boxes)


    # start get the label process.....
    grid_size = [1, 2, 4, 8]
    n_tile_strided = {}   # 1, 3, 7, 15
    for n_tile in grid_size:
        tile_size = 224 / n_tile
        stride = tile_size / 2
        n_tile_strided[n_tile] = np.floor((224 - stride) / stride)

    # hm store the label
    hm = {}
    for n_tile in grid_size:
        hm[n_tile] = np.zeros([int(n_tile_strided[n_tile]),
                               int(n_tile_strided[n_tile])])

    for GT_cnt in range(boxes.shape[0]):
        max_ov_allscale = -np.inf
        max_x_allscale = 0
        max_y_allscale = 0
        max_n_tile = 0
        has_ov_cell = False

        for n_tile in grid_size:
            tile_size = 224 / n_tile
            stride = tile_size / 2

            # top left belong
            cell_x_tl = np.floor((boxes[GT_cnt, 0] -1) / stride)
            cell_y_tl = np.floor((boxes[GT_cnt, 1] -1) / stride)

            # determine which cell the bot_right belonging to
            cell_x_br = np.floor((boxes[GT_cnt, 2] -1) / stride)
            cell_y_br = np.floor((boxes[GT_cnt, 3] -1) / stride)  # - 1

            cell_x_tl = min(max(1, cell_x_tl), n_tile_strided[n_tile])
            cell_y_tl = min(max(1, cell_y_tl), n_tile_strided[n_tile])
            cell_x_br = min(max(1, cell_x_br), n_tile_strided[n_tile])
            cell_y_br = min(max(1, cell_y_br), n_tile_strided[n_tile])

            max_ov = -np.inf
            max_x = 0
            max_y = 0
            for x in range(int(cell_x_tl), int(cell_x_br + 1)):
                for y in range(int(cell_y_tl), int(cell_y_br) + 1):
                    ov = IOU.IOU_V2(np.array([(x - 1) * stride+1,
                                           (y - 1) * stride+1,
                                           (x - 1) * stride + tile_size,
                                           (y - 1) * stride + tile_size]),
                                       np.array(boxes[GT_cnt, 0:4])
                                       )

                    if (ov > 0.3):
                        hm[n_tile][y - 1, x - 1] = 1  # ???   -1 !!! hm_temp
                        has_ov_cell = True
                        if n_tile == 1 :#or n_tile==2 or n_tile==4:
                            print (n_tile, img_path)

                    if (ov > max_ov):
                        max_ov = ov
                        max_x = x
                        max_y = y

            if (max_ov > max_ov_allscale):
                max_ov_allscale = max_ov
                max_n_tile = n_tile
                max_x_allscale = max_x
                max_y_allscale = max_y

        if (not has_ov_cell):  # 如果没有一个是大于0.3的，那就把最大的那个设置为1
            # print ('triger the max_ov_allscale process!', img_path,'GT is:', GT_cnt ,boxes.shape[0],max_ov_allscale,max_n_tile,tile_size,stride)
            if (max_ov_allscale > 0):
                hm[max_n_tile][max_y_allscale-1, max_x_allscale-1] = 1

    output = []
    cnt = 0
    for n_tile in grid_size:
        gap = int(n_tile_strided[n_tile] ** 2)
        output[cnt: cnt + gap] = np.reshape(hm[n_tile],gap, order='F') #hm[n_tile].reshape(gap)
        cnt = cnt + gap

    return img_resize, output




def combine_global(BB, global_prob, img_path, alpha=0.5):
    print (' global model added !')
    # BB 是 [x, y, x, y, score] 格式的多个检测框数据，
    # 但是存在两种框的尺寸的问题，faster rcnn 传递过来的是原图尺寸，这里使用的是在pading 之后的224尺寸
    # 以及matlab 代码使用的是网络原始得分进行操作，而这里是否需要调整为使用 softmax之后的得分？
    # 因为最后的计算 map 必须使用该得分,ok ,确认直接使用
    # 网络得分也是可以得，那这里不存在问题了
    cell_size = [1, 3, 7, 15]
    grid_size = [1, 2, 4, 8]

    img = Image.open(img_path)

    if img.size[1] >= img.size[0]:
        large_side = img.size[1]
    else:
        large_side = img.size[0]

    scale_factor = 224.0 / large_side

    bw = bh = large_side
    ow, oh = img.size

    # pading image
    if img.size[1] >= img.size[0]:
        pad_size = int((bw - ow) / 2)
    else:
        pad_size = int((bh - oh) / 2)

    # reshape the global_prob result
    hm = {}
    cnt = 0
    for j, n_tile in enumerate(grid_size):
        gap = int(cell_size[j] ** 2)
        hm[n_tile] = np.reshape(global_prob[cnt: cnt + gap], (cell_size[j], cell_size[j]) , order='F')
        cnt = cnt + gap

    # pading and resize the BB boxes
    BB_pad = BB.copy()
    # pading box
    if img.size[1] >= img.size[0]:
        BB_pad[:, 0] = BB_pad[:, 0] + pad_size
        BB_pad[:, 2] = BB_pad[:, 2] + pad_size
    else:
        BB_pad[:, 1] = BB_pad[:, 1] + pad_size   # should all be this
        BB_pad[:, 3] = BB_pad[:, 3] + pad_size   # y1, y2 + pad_size  这里的框不需要两边都加pad_size?? 为什么作者代码没有加？？因为他是x1y1wh格式

    # resize box
    BB_pad[:, 0:4] = BB_pad[:, 0:4] * scale_factor


    for GT_cnt in range(BB_pad.shape[0]):
        max_ov_allscale = -np.inf
        max_x_allscale = 0
        max_y_allscale = 0
        max_n_tile = 0

        for n_tile in grid_size:
            tile_size = 224 / n_tile
            stride = tile_size / 2

            # top left belong
            cell_x_tl = np.floor((BB_pad[GT_cnt, 0] -1 ) / stride)
            cell_y_tl = np.floor((BB_pad[GT_cnt, 1] -1 ) / stride)

            # determine which cell the bot_right belonging to
            cell_x_br = np.floor((BB_pad[GT_cnt, 2] -1 ) / stride)
            cell_y_br = np.floor((BB_pad[GT_cnt, 3] -1 ) / stride)  # - 1


            cell_x_tl = max(1, cell_x_tl)   # 这里不需要设置边框超出图像的限制了吗？
            cell_y_tl = max(1, cell_y_tl)
            cell_x_br = max(1, cell_x_br)
            cell_y_br = max(1, cell_y_br)

            max_ov = -np.inf
            max_x = 0
            max_y = 0
            for x in range(int(cell_x_tl), int(cell_x_br + 1)):
                for y in range(int(cell_y_tl), int(cell_y_br) + 1):
                    ov = IOU.IOU_V2(np.array([(x - 1) * stride+1,
                                           (y - 1) * stride+1,
                                           (x - 1) * stride + tile_size,
                                           (y - 1) * stride + tile_size]),
                                       np.array(BB_pad[GT_cnt, 0:4])
                                       )

                    if (ov > max_ov):
                        max_ov = ov
                        max_x = x
                        max_y = y

            if (max_ov > max_ov_allscale):
                max_ov_allscale = max_ov
                max_n_tile = n_tile
                max_x_allscale = max_x
                max_y_allscale = max_y
        # 每个框只调整一次，就是使用具有最大 IOU 的 n_tile 里面的那个 cell 的值来调整
        BB[GT_cnt, 4] = BB_pad[GT_cnt, 4] * (1 - alpha) + alpha * hm[max_n_tile][max_y_allscale-1, max_x_allscale-1]

    return BB





def plot_heatmap(prob, img_name, np_index=None, boxes=None,
                 IMG_PATH=None, TEST_RESULT_PATH=None):
    # IMG_PATH = '/nishome/zl/faster-rcnn/data/prison_datasets/JPEGImages/'
    # TEST_RESULT_PATH = '/nishome/zl/faster-rcnn/data/prison_datasets/'
    # prob = np.random.rand(284)
    cell_size = [1, 3, 7, 15]
    i = 0
    back_ground = Image.new('RGB', (2000, 500), color=(255,255,255))

    p_max = np.max(prob)
    p_min = np.min(prob)

    img = Image.open(os.path.join(IMG_PATH,img_name))
    img = img.resize((400, 400))
    back_ground.paste(img, (0 , 0)) # paste original image
    for j, side in enumerate(cell_size):
        arr = np.zeros((side, side))
        for w in range(side):
            for h in range(side):
                arr[h, w] = prob[i]
                i += 1
        plt.matshow(arr, cmap='jet' ,vmin=p_min, vmax=p_max)
        plt.colorbar()
        img_path = os.path.join(TEST_RESULT_PATH, str(j) + '.jpg')
        plt.savefig(img_path, dpi=200)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((400, 400))
        back_ground.paste(img, (400 * (j + 1), 0))


    img_path_new = os.path.join(TEST_RESULT_PATH, 'result_'+img_name )
    back_ground.save(img_path_new)
    back_ground.close()
    img.close()




if __name__ == '__main__':

    pass
