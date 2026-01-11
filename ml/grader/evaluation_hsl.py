import numpy as np
import math
import elkai
import re

from colormath.color_conversions import *
from colormath.color_objects import *
from colormath.color_diff import delta_e_cie2000


# /color_utils/color.py
def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)

def hsl_to_rgb(hsl_str):
    """
    Convert HSL string format "(H, S%, L%)" to RGB tuple (0-255).
    
    Args:
        hsl_str: String in format "(H, S%, L%)"
        
    Returns:
        Tuple of (R, G, B) values in range 0-255
    """
    # Extract H, S, L values
    match = re.match(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)', hsl_str.strip())
    if not match:
        raise ValueError(f"Invalid HSL format: {hsl_str}")
    
    h = int(match.group(1)) % 360
    s = int(match.group(2)) / 100.0
    l = int(match.group(3)) / 100.0
    
    # Convert HSL to RGB
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h / 360 + 1/3)
        g = hue_to_rgb(p, q, h / 360)
        b = hue_to_rgb(p, q, h / 360 - 1/3)
    
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def hsl_to_hex(hsl_str):
    """
    Convert HSL string format "(H, S%, L%)" to hex color code.
    
    Args:
        hsl_str: String in format "(H, S%, L%)"
        
    Returns:
        Hex color code string like "#RRGGBB"
    """
    r, g, b = hsl_to_rgb(hsl_str)
    return f"#{r:02x}{g:02x}{b:02x}"


class Color:
    def __init__(self, source, is_RGB = True, is_HSL = False):
        
        if is_HSL:
            # Convert HSL string to RGB first
            rgb_tuple = hsl_to_rgb(source)
            self.rgb = sRGBColor(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2], is_upscaled=True)
            self.hex = self.rgb.get_rgb_hex()
        elif is_RGB:
            self.rgb = sRGBColor(source[0], source[1], source[2], is_upscaled=True)
            self.hex = self.rgb.get_rgb_hex()
        else:
            self.rgb = sRGBColor.new_from_rgb_hex(source)
            self.hex = source
        
        self.lab = convert_color(self.rgb, LabColor)

    ###
    #   Color Representation Getters
    ###
    
    def RGB(self):
        return self.rgb
    
    def HEX(self):
        return self.hex
    
    def LAB(self):
        return self.lab
    
    def LAB_val(self):
        return self.lab.get_value_tuple()
    
    def LAB_geo(self):
        L, A, B = self.LAB_val()

        x = A / 256
        y = B / 256
        z = (L / 100) - 0.5

        return (x, y, z)

    def __deepcopy__(self, memo):
        new_color = Color(self.hex, is_RGB=False)
        memo[id(self)] = new_color
        return new_color
    
    def __str__(self):
        return self.HEX()
#####



# /color_utils/color_palette.py
class ColorPalette:
    def __init__(self, source, option='rgb', original_palette_group = None):
        """
            option = <'rgb' | 'color' | 'hex' | 'hsl'>
        """
        self.colors = []
        
        if option == 'rgb':
            for s in source:
                self.colors.append(Color(s, is_RGB=True))
        elif option == 'hex':
            for s in source:
                self.colors.append(Color(s, is_RGB=False))
        elif option == 'hsl':
            for s in source:
                self.colors.append(Color(s, is_RGB=False, is_HSL=True))
        elif option == 'color':
            for s in source:
                self.colors.append(copy.deepcopy(s))
        else:
            raise ValueError("option must be 'rgb', 'color', 'hex', or 'hsl'")

        self.original_palette_group = original_palette_group


    def to_hex_list(self, order=None):
        # return shape: ['#0e638d', '#7ba9a0', '#e6d6cf', '#e3a07f']
        if order:
            return self._get_HEX_color_objects(order)
        else:
            return [c.HEX() for c in self.colors]

    def to_serialized_hex_string(self):
        return '\n'.join(self.to_hex_list())
    
    def length(self):
        return len(self.colors)
    
    def calculate_palette_diversity(self):
        colors = self.get_LAB_color_objects()
        pairwise_distance = 0.0
        num_pairs = 0
        N = len(colors)
        for i in range(N - 1):
            cur_color = colors[i]
            distances = [delta_e_cie2000(cur_color, colors[j]) for j in range(i+1, N)]
            num_pairs += len(distances)
            pairwise_distance += sum(distances)
        
        return pairwise_distance / num_pairs if num_pairs > 0 else 0.0

    ###
    #   Graph Length Helpers
    ###

    def get_graph_length_in_order(self, order):
        return sum(self.get_graph_length_list_in_order(order))
        
    def get_graph_length_list_in_order(self, order):
        colors = self._get_LAB_color_objects(order)
        lengths = []
        for i in range(len(colors) - 1):
            cur_color = colors[i]
            next_color = colors[i+1]
            # gets distance between two color's lab representations
            distance = delta_e_cie2000(cur_color, next_color)

            lengths.append(distance)

        return lengths

    ###
    #   Color Space Getters
    ###

    def get_LAB_color_objects(self, order=None):
        if order is None:
            order = list(range(self.length()))
        
        return self._get_LAB_color_objects(order)
    
    def get_LAB_color_values(self, order=None, is_geo=False):
        if order is None:
            order = list(range(self.length()))
        
        return self._get_LAB_color_values(order, is_geo)

    def get_original_palette_group(self):
        return self.original_palette_group
        

    ###
    #   Internal Getters
    ###

    def _get_LAB_color_objects(self, order):
        result = []
        for idx in order:
            result.append(self.colors[idx].LAB())
        return result

    def _get_LAB_color_values(self, order, is_geo=False):
        result = []
        for idx in order:
            if not is_geo:
                result.append(self.colors[idx].LAB_val())
            else:
                result.append(self.colors[idx].LAB_geo())
        return result

    def _get_HEX_color_objects(self, order):
        result = []
        for idx in order:
            result.append(self.colors[idx].HEX())
        return result

    @staticmethod
    def merge_to_single_palette(palettes):
        merged_hex_list = []
        original_palette_group = []

        for index, palette in enumerate(palettes):
            merged_hex_list += palette.to_hex_list()
            original_palette_group += [index for i in range(palette.length())]

        return ColorPalette(source=merged_hex_list, option='hex', original_palette_group=original_palette_group)

    def __str__(self):
        return ' '.join(self.to_hex_list())
#####



# /color_utils/single_palette_sorter.py
class SinglePaletteSorter:
	def __init__(self, palette):
		self.palette = palette
    

	def sort(self, with_cutting=False, debug=False):
		sorted_indices = self._tsp_sort(with_cutting)

		if debug:
			print("--- SPS results")
			print("sorted_indicies: ", sorted_indices)
			print("sorted hex list: ", self.palette.to_hex_list(order=sorted_indices))

		return sorted_indices


	def _tsp_sort(self, with_cutting=False):
		"""
			default SPS
			uses ciede2000 and lkh tsp solver
		"""
		distance_matrix = self._get_distance_matrix()
		enlarged_distance_matrix = [[round(e*10000) for e in dm] for dm in distance_matrix]

		best_state = elkai.solve_int_matrix(enlarged_distance_matrix, 100)

		reordered_best_state = None
		direction, initial_node_index = self._TSP_graph_cut(best_state, distance_matrix, with_cutting)

		if direction == 'Forward':
			reordered_best_state = best_state

		elif direction == 'Reverse':
			best_state.reverse()
			reordered_best_state = best_state

		else:
			assert False, '[_tsp_sort] No such graph cut direction'

		# reorder color palette
		result = reordered_best_state[reordered_best_state.index(initial_node_index):] + reordered_best_state[:reordered_best_state.index(initial_node_index)]
		color_values = self.palette.get_LAB_color_values(is_geo=True)

		return result


    ###
    #   Travelling Sales Man Problem (TSP) Helpers
    ###

	def _get_distance_matrix(self):
		color_list = self.palette.get_LAB_color_objects()

		distance_matrix = []
		for color_a in color_list:
			sub_distance_matrix = []

			for color_b in color_list:
				distance = delta_e_cie2000(color_a, color_b)
				sub_distance_matrix.append(distance)

			distance_matrix.append(sub_distance_matrix)

		return distance_matrix
    

	def _TSP_graph_cut(self, TSP_graph_indices, distance_matrix, with_cutting=False):
		start_index = None
		end_index = None
		
		if with_cutting:
			start_index, end_index = self._TSP_cut_palette_largest(TSP_graph_indices, distance_matrix)
		else:
			start_index, end_index = self._TSP_cut_whole_largest(TSP_graph_indices, distance_matrix)
		
		# direction decision: compare two tips
		# Regardless of the direction, initial_node_index indicates the index of first element in the sorted result.
		direction = None
		initial_node_index = None

		lightness_costs = self._get_lightness_costs()
		if lightness_costs[start_index] >= lightness_costs[end_index]:
			initial_node_index = start_index
			direction = 'Reverse'
		else:
			initial_node_index = end_index
			direction = 'Forward'

		# direction: <'Forward' | 'Reverse'>
		return direction, initial_node_index


	def _TSP_cut_whole_largest(self, TSP_graph_indices, distance_matrix):
		max_distance = - math.inf
		start_index = None
		end_index = None

		neighbor_pairs_indices = [((i), (i + 1) % len(TSP_graph_indices)) for i in range(len(TSP_graph_indices))] 
		neighbor_pairs = [(TSP_graph_indices[i], TSP_graph_indices[j]) for i, j in neighbor_pairs_indices]

		for neighbor_pair_start, neighbor_pair_end  in neighbor_pairs:
			distance = distance_matrix[neighbor_pair_start][neighbor_pair_end]
			if max_distance < distance:
				max_distance = distance
				start_index = neighbor_pair_start
				end_index = neighbor_pair_end

		return start_index, end_index
    
    
	def _TSP_cut_palette_largest(self, TSP_graph_indices, distance_matrix):
		# When we should cut between A and B located in forward direction, then A becomes start_idnex and B does end_index.
		# Cutting assumed
		start_index = None
		end_index = None

		largest_palette_start, largest_palette_end = self._find_largest_palette_indices(TSP_graph_indices, distance_matrix)
		# Decide the target: inner or outer colors
		smaller_boundary = min(TSP_graph_indices.index(largest_palette_start), TSP_graph_indices.index(largest_palette_end))
		larger_boundary = max(TSP_graph_indices.index(largest_palette_start), TSP_graph_indices.index(largest_palette_end))

		is_outer, target_colors = self._find_colors_inside_boundary(TSP_graph_indices, larger_boundary, smaller_boundary)

		if len(target_colors) == 0:
			start_index = largest_palette_start
			end_index = largest_palette_end

		else:
			if is_outer:
				start_index, end_index = self._find_outer_case_start_end_indices(TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary)
			else:
				start_index, end_index = self._find_inner_case_start_end_indices(TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary)

		return start_index, end_index
    
    
	def _find_largest_palette_indices(self, TSP_graph_indices, distance_matrix):
		if self.palette.get_original_palette_group() is None:
			raise AttributeError("Attribute 'original_palette_group' is None. Are you sure your currently palette is properly merged?")
		
		max_distance = - math.inf
		original_palette_group = self.palette.get_original_palette_group()
		largest_palette_start = None
		largest_palette_end = None

		for group_no in range(max(original_palette_group)+1):
			mask_list = [original_palette_group[k] == group_no for k in TSP_graph_indices]
			filtered_list = [i for (i, v) in zip(TSP_graph_indices, mask_list) if v]

			neighbor_pairs_indices = [((i), (i + 1) % len(filtered_list)) for i in range(len(filtered_list))] 
			neighbor_pairs = [(filtered_list[i], filtered_list[j]) for i, j in neighbor_pairs_indices]

			for neighbor_pair_start, neighbor_pair_end  in neighbor_pairs:
				distance = distance_matrix[neighbor_pair_start][neighbor_pair_end]
				if max_distance < distance:
					max_distance = distance
					largest_palette_start = neighbor_pair_start
					largest_palette_end = neighbor_pair_end
					
		return largest_palette_start, largest_palette_end
    
    
	def _find_colors_inside_boundary(self, TSP_graph_indices, larger_boundary, smaller_boundary):
		is_outer = False
		target_colors = []
		if larger_boundary - smaller_boundary > self.palette.length() * 0.5:
			# outer colors: [, smaller_boundary-1] and [larger_boundary+1, ]
			is_outer = True

			for i in range(smaller_boundary):
				target_colors.append(TSP_graph_indices[i])

			for i in range(larger_boundary+1, self.palette.length()):
				target_colors.append(TSP_graph_indices[i])

		else:
			# inner colors: [smaller_boundary+1, larger_boundary-1]
			for i in range(smaller_boundary+1, larger_boundary):
				target_colors.append(TSP_graph_indices[i])

		return is_outer, target_colors


	def _find_outer_case_start_end_indices(self, TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary):
		# Cutting assumed
		larger_index = TSP_graph_indices[larger_boundary]
		smaller_index = TSP_graph_indices[smaller_boundary]
		start_index = target_colors[-1]
		end_index = smaller_index

		distance_from_smaller_boundary = []
		distance_from_larger_boundary = []

		for i in range(len(target_colors)):
			cur_index = TSP_graph_indices[(larger_boundary + 1 + i) % len(TSP_graph_indices)]
			distance_from_smaller_boundary.append(distance_matrix[smaller_index][cur_index])
			distance_from_larger_boundary.append(distance_matrix[larger_index][cur_index])

		min_distance_sum = math.inf			
		for i in range(len(distance_from_smaller_boundary)+1):
			cur_distance_sum = sum(distance_from_larger_boundary[:i]) + sum(distance_from_smaller_boundary[i:])

			if min_distance_sum > cur_distance_sum:
				start_index = TSP_graph_indices[(larger_boundary + i) % len(TSP_graph_indices)]
				end_index = TSP_graph_indices[(larger_boundary + i + 1) % len(TSP_graph_indices)]
				min_distance_sum = cur_distance_sum

			
		return start_index, end_index


	def _find_inner_case_start_end_indices(self, TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary):
		larger_index = TSP_graph_indices[larger_boundary]
		smaller_index = TSP_graph_indices[smaller_boundary]
		start_index = target_colors[-1]
		end_index = larger_index

		distance_from_smaller_boundary = []
		distance_from_larger_boundary = []

		for i in range(smaller_boundary + 1, larger_boundary):
			cur_index = TSP_graph_indices[i]
			distance_from_smaller_boundary.append(distance_matrix[smaller_index][cur_index])
			distance_from_larger_boundary.append(distance_matrix[larger_index][cur_index])

		min_distance_sum = math.inf			
		for i in range(len(distance_from_smaller_boundary)+1):
			cur_distance_sum = sum(distance_from_smaller_boundary[:i]) + sum(distance_from_larger_boundary[i:])

			if min_distance_sum > cur_distance_sum:
				start_index = TSP_graph_indices[smaller_boundary + i]
				end_index = TSP_graph_indices[smaller_boundary + i + 1]
				min_distance_sum = cur_distance_sum

		return start_index, end_index


	def _get_lightness_costs(self):
		# White has the lowest value 0 and black has the highest value 1
		lightness_costs = []
		geo_coords = self.palette.get_LAB_color_values(is_geo=True)
		lightness_costs = [0.5 - geo_coord[2] for geo_coord in geo_coords]

		return lightness_costs
#####



# /color_utils/multiple_palettes_sorter.py
class MultiplePalettesSorter:
    def __init__(self, palettes):
        self.palettes = [ColorPalette(p.to_hex_list(), option='hex') for p in palettes if p is not None] # list of ColorPalette
        self.palette_count = len(palettes)
    
    
    def sort_merge(self, with_cutting=True, debug=False):
        sorted_result = []

        # 1. merge palette
        merged_palettes = ColorPalette.merge_to_single_palette(self.palettes)

        # 2. joint sort 
        single_palette_sorter = SinglePaletteSorter(merged_palettes)
        merged_sorted_indices = single_palette_sorter.sort(with_cutting, debug)

        # 3. Re-distribute the result
        color_start_index = 0
        color_end_index = 0
        for p_index in range(self.get_palettes_count()):
            color_end_index += self.palettes[p_index].length()
            sorted_result.append([i - color_start_index for i in merged_sorted_indices if i >= color_start_index and i < color_end_index])
            color_start_index = color_end_index

        merged_length = merged_palettes.get_graph_length_in_order(merged_sorted_indices)

        return sorted_result, merged_length, merged_sorted_indices
    
    def get_hex_lists(self):
        return [p.to_hex_list() for p in self.palettes]
    
    def get_palette_objects(self):
        return self.palettes
    
    def get_palettes_count(self):
        return self.palette_count
#####



# /color_utils/dccw_measurer.py
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
#####
import os
import pytest
import logging
from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

logger = logging.getLogger(__name__)

def normalize_inv_map(error_val, tau, k=1):
    """
    Maps an error to a score [0, 1] (less error_val -> closer to 1)
    tau: The error value at which the score should be 0.5
    """
    p = k * (error_val - tau)
    return 1.0 / (1.0 + math.exp(p))

def harmonic_mean(score_a, score_b):
    """
    Combine two scores using their harmonic mean
    """
    ep = 1e-6
    
    return (2 * score_a * score_b) / (score_a + score_b + ep)


def extract_color_palette(text: str)->list[str] | None:
    """
    Extracts HSL color palette from text
    Performs Format compliance checking for HSL format (H, S%, L%)
    
    Args:
        text: text containing the generated color palette in HSL format
    
    Returns:
        A list of HSL strings if text is in valid format
        None if text is not format compliant or does not contain an extractable color palette
    """
    
    hsl_pattern = r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)'
    matches = re.findall(hsl_pattern, text)
    
    # Return first 5 unique colors in HSL format
    seen = []
    for match in matches:
        hsl_str = f"({match[0]}, {match[1]}%, {match[2]}%)"
        if hsl_str not in seen:
            seen.append(hsl_str)
        if len(seen) >= 5:
            break
    
    if len(seen) < 5:
        return None
    return seen


JSONL_PATH = os.path.abspath("./ml/grader/development/chromio_v0_hsl.jsonl")

@evaluation_test(
    input_dataset=[JSONL_PATH],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/qwen3-8b"}], # lowkey higher temp maybe?
    # max_dataset_rows=5, # TODO: Remove during training
    passed_threshold=0.6, # threshold for average R_score accross entire dataset
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
async def evaluator(row: EvaluationRow, **kwargs)->EvaluationRow:
    """
    Evaluate generated color palette (in HSL format) considering both similarity to ground truths palette and similar diversity
    
    This Function utilizes multiple evaluation criteria:
    - Format compliance checking for output palette in HSL format (H, S%, L%)
    - Combines both similarity score (S) and diversity score (D) and calcualte their harmonic mean

    Args:
        row: EvaluationRow containing the conversation messages, ground truth palette, and ground truth palette diversity (Optional)
        **kwargs: Additional hyperparameters like the ones for normalization
    
    Returns:
        EvaluationRow with evaluation_result
    """
    logger.info(f"I am beginning to execute Chromio rollout: {row.execution_metadata.rollout_id}")

    
    D_norm_kwargs = kwargs.get('D_norm_kwargs', {"tau": 15.0, "k": 0.2})
    S_norm_kwargs = kwargs.get('S_norm_kwargs', {"tau": 26.0, "k": 0.15})
    
    assistant_messages = [m for m in row.messages if getattr(m, "role", None) == "assistant"]
    last_assistant_content = assistant_messages[-1].content if assistant_messages and getattr(assistant_messages[-1], "content", None) else ""
    
    prediction_palette = extract_color_palette(str(last_assistant_content))
    if prediction_palette is None or not hasattr(row, 'gt_palette'):
        is_score_valid = False
        reason="Unknown Reason."
        if prediction_palette is None:
            reason = "Invalid Model Output Format: Cannot find prediction palette in HSL format"
            is_score_valid = True # punish incorrect format generation
        elif not hasattr(row, 'gt_palette'):
            reason = "Invalid Data: Missing gt_palette"
        
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=is_score_valid,
            reason=reason
        )
        return row


    gt_palette = row.gt_palette
    dccw_measurer = DccwMeasurer(
        source=prediction_palette,
        source_option='hsl',
        target=gt_palette,
        target_option='hsl'
    )
    
    dccw_score = dccw_measurer.measure_dccw(reflect_cycle=False)
    diversity = dccw_measurer.calculate_source_diversity()
    gt_diversity = float(row.gt_diversity) if hasattr(row, 'gt_diversity') else dccw_measurer.calculate_target_diversity()
    
    norm_D = normalize_inv_map(abs(gt_diversity - diversity), tau=D_norm_kwargs['tau'], k=D_norm_kwargs['k'])
    norm_S = normalize_inv_map(dccw_score, tau=S_norm_kwargs['tau'], k=S_norm_kwargs['k'])
    score_R = harmonic_mean(norm_D, norm_S)
    reason = f"Prediction: {dccw_measurer.get_source_HEX_after_sort()}, Ground Truths: {dccw_measurer.get_target_HEX_after_sort()}\t| norm_Diversity: {norm_D}, norm_DCCW: {norm_S}"

    evaluation_result = EvaluateResult(
        score=score_R,
        is_score_valid=True,
        reason=reason
    )
    
    logger.info(f"I am done executing Chromio rollout: {row.execution_metadata.rollout_id}")
    row.evaluation_result = evaluation_result
    return row
    
    

if __name__ == "__main__":
    
    # temporary test with HSL colors
    source_hsl_list = ["(0, 85%, 46%)", "(240, 57%, 38%)", "(267, 32%, 50%)", "(240, 34%, 38%)", "(13, 59%, 32%)"]
    target_hsl_list = ["(45, 41%, 27%)", "(11, 30%, 17%)", "(75, 43%, 17%)", "(270, 11%, 38%)", "(240, 2%, 21%)"]
    
    dccw_measurer = DccwMeasurer(
        source=source_hsl_list,
        source_option='hsl',
        target=target_hsl_list,
        target_option='hsl'
    )
    
    dccw_score_no_cycle = dccw_measurer.measure_dccw(reflect_cycle=False)
    dccw_score_with_cycle = dccw_measurer.measure_dccw(reflect_cycle=True)
    
    source_diversity = dccw_measurer.calculate_source_diversity()
    target_diversity = dccw_measurer.calculate_target_diversity()
    
    print("-------------------------------")
    print("Source Palette Diversity: ", source_diversity)
    print("Target Palette Diversity: ", target_diversity)
    print("-------------------------------")
    print("DCCW score (no cycle): ", dccw_score_no_cycle)
    print("DCCW score (with cycle): ", dccw_score_with_cycle)
    print("-------------------------------")
    
    norm_D = normalize_inv_map(abs(target_diversity - source_diversity), tau=15.0, k=0.2)
    norm_S = normalize_inv_map(dccw_score_no_cycle, tau=26.0, k=0.15)
    score_R = harmonic_mean(norm_D, norm_S)
    
    print("Normalized Diversity diff: ", norm_D)
    print("Normalized DCCW: ", norm_S)
    print("Final Harmonic Mean (Grader Score): ", score_R)
    print("-------------------------------")
