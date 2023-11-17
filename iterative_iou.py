from mmdet3d.structures import BaseInstance3DBoxes,

def iterated_iou(boxes_list: [BaseInstance3DBoxes],) -> [dict]:
    box_num = len(boxes_list)
    boxes_list = [dict(key_region=box.bev[...,:4], other_area=box.dims[...,0] * box.dims[...,1],
                       iou_calculated=None, iou_matrix=None, meta_shape=(box.shape[0],)) for box in boxes_list]

    former_boxes = boxes_list[0]
    for n in range(1, box_num):
        later_boxes = boxes_list[n]
        former_boxes = single_iou(former_boxes, later_boxes)

    return former_boxes

def single_iou(former_boxes:dict, later_boxes: dict) -> torch.tensor:
    """
    args:
        input: dict
            key_region: numpy.array.
                shape: (M, 4)
            other_area: numpy.array.
                shape: (M, )
            iou_calculated: bool.
            iou_matrices: torch.tensor.
        new_boxes: dict
            key_region: numpy.array.
                shape: (N, 4)
            other_area: numpy.array.
                shape: (N, )
            iou_calculated: bool
            iou_matrices: torch.tensor.
    return:
        iou_matrices: dict
            key_region: numpy.array.
                shape: (N, 4)
            other_area: numpy.array.
                shape: (N, )
            iou_calculated: bool
            iou_matrices: torch.tensor.

    """
    meta_shape = former_boxes['meta_shape']
    key_region = former_boxes['key_region'].reshape(np.prod(meta_shape), 4)
    other_area_f = former_boxes['other_area'].reshape(np.prod(meta_shape))

    new_boxes = later_boxes['key_region']
    other_area_l = later_boxes['other_area']

    M = len(key_region)
    N = len(new_boxes)

    kr_x1, kr_x2 = key_region[..., 0] - key_region[..., 2] / 2, key_region[..., 0] + key_region[..., 2] / 2
    kr_y1, kr_y2 = key_region[..., 1] - key_region[..., 3] / 2, key_region[..., 1] + key_region[..., 3] / 2
    kr = torch.zeros((key_region.shape[0],4), device=key_region.device, dtype=key_region.dtype)
    kr[..., 0], kr[...,1], kr[...,2], kr[...,3] = kr_x1, kr_y1, kr_x2, kr_y2

    nb_x1, nb_x2 = new_boxes[..., 0] - new_boxes[..., 2] / 2, new_boxes[..., 0] + new_boxes[..., 2] / 2
    nb_y1, nb_y2 = new_boxes[..., 1] - new_boxes[..., 3] / 2, new_boxes[..., 1] + new_boxes[..., 3] / 2
    nb = torch.zeros((new_boxes.shape[0],4), device=new_boxes.device, dtype=new_boxes.dtype)
    nb[..., 0], nb[...,1], nb[...,2], nb[...,3] = nb_x1, nb_y1, nb_x2, nb_y2

    # iou_matrices = torch.zeros((M, N), device=key_region.device, dtype=key_region.dtype)
    max_xy = torch.min(kr[..., 2:].unsqueeze(1).expand(M, N, 2),
                       nb[..., 2:].unsqueeze(0).expand(M, N, 2))
    min_xy = torch.max(kr[..., :2].unsqueeze(1).expand(M, N, 2),
                          nb[..., :2].unsqueeze(0).expand(M, N, 2))
    inter = torch.clamp((max_xy - min_xy), min=0) ## (M, N, 2)
    inter = inter[..., 0] * inter[..., 1] ## (M, N)

    # kr_area = (kr[..., 2] - kr[..., 0]) * (kr[..., 3] - kr[..., 1])
    # nb_area = (nb[..., 2] - nb[..., 0]) * (nb[..., 3] - nb[..., 1]) ## (N, )

    other_area_new = other_area_f.unsqueeze(1).expand_as(inter) + other_area_l.unsqueeze(0).expand_as(inter) - inter
    iou_matrices = inter / other_area_new
    meta_shape = [*meta_shape, N]
    iou_matrices = iou_matrices
    # key_region_new = torch.cat([min_xy, max_xy], dim=-1)
    wh = max_xy - min_xy
    xy = max_xy - wh / 2
    key_region_new = torch.cat([xy, wh], dim=-1)
    return dict(key_region=key_region_new.reshape(*meta_shape, 4), other_area=other_area_new.reshape(meta_shape),
                iou_calculated=True,
                iou_matrix=iou_matrices.reshape(meta_shape), meta_shape=meta_shape)
