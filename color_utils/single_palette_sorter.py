import math
import enum
import numpy as np
import elkai
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
			print("with_cutting=True TODO...")
			pass
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
    

	def _get_lightness_costs(self):
		# White has the lowest value 0 and black has the highest value 1
		lightness_costs = []
		geo_coords = self.palette.get_LAB_color_values(is_geo=True)
		lightness_costs = [0.5 - geo_coord[2] for geo_coord in geo_coords]

		return lightness_costs