import numpy as np
from numba import jit


@jit(["int32(int32, int32, int32)"])
def calc_IOU1(img_shape, bbox_1, bbox_2):
	for _ in range(0, 5000):
		mask = np.zeros(shape=img_shape)
		mask[bbox_1[1]:bbox_1[1] + bbox_1[3], bbox_1[0]:bbox_1[0] + bbox_1[2]] += 1
		mask[bbox_2[1]:bbox_2[1] + bbox_2[3], bbox_2[0]:bbox_2[0] + bbox_2[2]] += 1
		pt1 = mask[bbox_1[1]:bbox_1[1] + bbox_1[3], bbox_1[0]:bbox_1[0] + bbox_1[2]] == 2
		count_of_mask_1_shared = np.count_nonzero(pt1)
		pt2 = mask[bbox_1[1]:bbox_1[1] + bbox_1[3], bbox_1[0]:bbox_1[0] + bbox_1[2]] > 0
		total_mask_1 = np.count_nonzero(pt2)
		pt3 = mask[bbox_2[1]:bbox_2[1] + bbox_2[3], bbox_2[0]:bbox_2[0] + bbox_2[2]] == 2
		count_of_mask_2_shared = np.count_nonzero(pt3)
		pt4 = mask[bbox_2[1]:bbox_2[1] + bbox_2[3], bbox_2[0]:bbox_2[0] + bbox_2[2]] > 0
		total_mask_2 = np.count_nonzero(pt4)
		t = max(count_of_mask_1_shared / total_mask_1, count_of_mask_2_shared / total_mask_2)
	
	return 1.0


if __name__ == "__main__":
	bb1 = (1814, 284, 50, 50)
	bb2 = (1941, 241, 50, 50)
	t = calc_IOU1((1700, 2560), bb1, bb2)
	print("Finished")
