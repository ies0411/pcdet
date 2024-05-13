import numpy as np


# class score, x, y, z, dx,dy,dz)
def iou(box_a, box_b):
    """
    deprecated function -> substituted to cpp
    """
    box_a_top_right_corner = [box_a[1] + (box_a[4] / 2.0), box_a[2] + (box_a[5] / 2.0)]
    box_b_top_right_corner = [box_b[1] + (box_b[4] / 2.0), box_b[2] + (box_b[5] / 2.0)]

    box_a_area = (box_a[4]) * (box_a[5])
    box_b_area = (box_b[4]) * (box_b[5])

    if box_a_top_right_corner[0] < box_b_top_right_corner[0]:
        length_xi = box_a_top_right_corner[0] - (box_b[1] - (box_b[4] / 2.0))
    else:
        length_xi = box_b_top_right_corner[0] - (box_a[1] - (box_a[4] / 2.0))

    if box_a_top_right_corner[1] < box_b_top_right_corner[1]:
        length_yi = box_a_top_right_corner[1] - (box_b[2] - (box_b[5] / 2.0))
    else:
        length_yi = box_b_top_right_corner[1] - (box_a[2] - (box_a[5] / 2.0))

    intersection_area = length_xi * length_yi

    if length_xi <= 0 or length_yi <= 0:
        iou = 0
    else:
        iou = intersection_area / (box_a_area + box_b_area - intersection_area)
    return iou


def nms(
    original_boxes,
    iou_thres_same_class=0.3,
    # iou_thres_different_class=0.6,
):
    # print(original_boxes)
    np_boxes = []
    for original_box in original_boxes:
        np_boxes.append(np.array(original_box))
    np_boxes = np.array(np_boxes)
    original_boxes = np_boxes[:, :-1]
    # print(type(original_boxes))
    original_boxes = np.array(original_boxes)
    boxes_probability_sorted = original_boxes[
        np.flip(np.argsort(original_boxes[:, -1]))
    ]
    # print(f"boxes_probability_sorted : {boxes_probability_sorted}")
    selected_boxes = []
    for idx, bbox in enumerate(boxes_probability_sorted):
        if bbox[-1] > 0:
            bbox_list = bbox.tolist()
            bbox_list.append(0.5)
            selected_boxes.append(bbox_list)
            for idx_2, other_box in enumerate(boxes_probability_sorted):
                if (
                    iou(bbox[:-1], other_box[:-1]) > iou_thres_same_class
                    and idx != idx_2
                ):
                    other_box[-1] = 0
    # print(f"selected_boxes : {selected_boxes}")
    return selected_boxes


# x,y,z,dx,dy,dz,yaw,score

# score,x,y,z,dx,dy,dz,yaw,class,idx
