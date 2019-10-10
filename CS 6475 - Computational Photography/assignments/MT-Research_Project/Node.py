import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


class Node:
    d_val = [1, 0]
    l_val = [1, -1]
    r_val = [1, 1]
    
    def __init__(self, location, image_shape, previous=None,
                 row=None, column=None):
        self.location = location
        self.image_shape = image_shape
        self.left_val = None
        self.right_val = None
        self.down_val = None
        self.neighbor_locations = None
        self.previous = previous
        self.finished = False
        self.calculate_neighbor_locations()
        
    def calculate_neighbor_locations(self):
        if self.previous is None:
            self.previous = self.location
        temp_locations = np.empty(3, dtype=np.ndarray)
        temp_locations[0] = self.get_left()
        self.left_val = temp_locations[0]
        temp_locations[1] = self.get_right()
        self.right_val = temp_locations[1]
        temp_locations[2] = self.get_down()
        self.down_val = temp_locations[2]
        self.neighbor_locations = temp_locations
        if self.neighbor_locations is None:
            print("END")
        return
        
    def get_down(self):
        down_loc = np.asarray([self.location[0] + self.d_val[0],  self.location[1] + self.d_val[1]])
        if self.in_boundaries(down_loc):
            return down_loc
        else:
            return None
        
    def get_left(self):
        left_loc = np.asarray([self.location[0] + self.l_val[0],  self.location[1] + self.l_val[1]])
        if self.in_boundaries(left_loc):
            return left_loc
        else:
            return None
        
    def get_right(self):
        right_loc = np.asarray([self.location[0] + self.r_val[0],  self.location[1] + self.r_val[1]])
        if self.in_boundaries(right_loc):
            return right_loc
        else:
            return None
        
    def in_boundaries(self, test_loc):
        test_row = test_loc[0]
        test_column = test_loc[1]
        if test_row in range(self.image_shape[0]) and test_column in range(self.image_shape[1]):
            if self.previous is not None:
                if np.all(self.previous == test_loc):
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False
        
    def __str__(self):
        return "The node location: {} \n" \
               "The Left neighbor: {} \n" \
               "The Right neighbor: {} \n" \
               "The Down neighbor: {} \n".format(self.location,
                                                 self.neighbor_locations[0],
                                                 self.neighbor_locations[1],
                                                 self.neighbor_locations[2])
