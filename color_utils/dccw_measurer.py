import numpy as np
import math
from scipy.spatial.distance import *

from color_utils.multiple_palettes_sorter import *
from color_utils.color_palette import ColorPalette

class DccwMeasurer:
    def __init__(self, source, source_option, target, target_option):
        self.source_palette = ColorPalette(source, option=source_option)
        self.target_palette = ColorPalette(target, option=target_option)
        
        multiple_palettes_sorter = MultiplePalettesSorter([self.source_palette, self.target_palette])
        self.sorted_palette_indices, _, _ = multiple_palettes_sorter.sort_merge()

    
    def get_palette_sorted_indices(self):
        return self.sorted_palette_indices
    

    def get_source_HEX_before_sort(self):
        return self.source_palette.to_hex_list()

    def get_target_HEX_before_sort(self):
        return self.target_palette.to_hex_list()
    
    def get_source_HEX_after_sort(self):
        return self.source_palette.to_hex_list(order=self.sorted_palette_indices[0])

    def get_target_HEX_after_sort(self):
        return self.target_palette.to_hex_list(order=self.sorted_palette_indices[1])
    
    def calculate_source_diversity(self):
        return self.source_palette.calculate_palette_diversity()

    def calculate_target_diversity(self):
        return self.target_palette.calculate_palette_diversity()
    

    def measure_dccw(self, reflect_cycle=False):
        source_labs = self.source_palette.get_LAB_color_values(order=self.sorted_palette_indices[0])
        target_labs = self.target_palette.get_LAB_color_values(order=self.sorted_palette_indices[1])

        distance_s_t, count_s_t, _ = self._dccw_from_A_to_B(source_labs, target_labs, reflect_cycle)
        distance_t_s, count_t_s, _ = self._dccw_from_A_to_B(target_labs, source_labs, reflect_cycle)
        
        return (distance_s_t + distance_t_s) / (count_s_t + count_t_s)


    def _dccw_from_A_to_B(self, A_colors, B_colors, reflect_cycle):
        distance = 0
        closest_points = []
        for a in A_colors:
            d, closest_point = self._dccw_from_a_to_B(a, B_colors, reflect_cycle)
            distance += d
            closest_points.append(closest_point)

        return distance, len(A_colors), closest_points
	

    def _dccw_from_a_to_B(self, a_color, B_colors, reflect_cycle):
        min_distance = math.inf
        min_closest_point = None

        color_range = len(B_colors)-1
        if reflect_cycle:
            color_range = len(B_colors)

        for b_index in range(color_range):
            b_segment_start = np.array(B_colors[b_index])
            b_segment_end = np.array(B_colors[(b_index+1) % len(B_colors)])

            a = np.array(a_color)

            distance, closest_point = self._point_to_line_dist(a, b_segment_start, b_segment_end)
            if distance < min_distance:
                min_distance = distance
                min_closest_point = closest_point

        return min_distance, min_closest_point


    def _point_to_line_dist(self, p, a, b):
        # https://stackoverflow.com/a/44129897/3923340
        # project c onto line spanned by a,b but consider the end points should the projection fall "outside" of the segment    
        n, v = b - a, p - a

        # the projection q of c onto the infinite line defined by points a,b
        # can be parametrized as q = a + t*(b - a). In terms of dot-products,
        # the coefficient t is (c - a).(b - a)/( (b-a).(b-a) ). If we want
        # to restrict the "projected" point to belong to the finite segment
        # connecting points a and b, it's sufficient to "clip" it into
        # interval [0,1] - 0 corresponds to a, 1 corresponds to b.

        t = max(0, min(np.dot(v, n)/np.dot(n, n), 1))
        closest_point = (a + t*n)
        distance = np.linalg.norm(p - closest_point) #or np.linalg.norm(v - t*n)

        return distance, closest_point