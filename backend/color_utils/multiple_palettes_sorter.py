from color_utils.single_palette_sorter import *
from color_utils.color_palette import ColorPalette

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