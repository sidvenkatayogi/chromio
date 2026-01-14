import math
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from colormath.color_diff import delta_e_cie2000

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

		best_state, _ = solve_tsp_dynamic_programming(np.array(enlarged_distance_matrix))

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