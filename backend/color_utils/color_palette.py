from color_utils.color import Color
from colormath.color_diff import delta_e_cie2000

class ColorPalette:
    def __init__(self, source, option='rbg', original_palette_group = None):
        """
            option = <'rbg' | 'color' | 'hex'>
        """
        self.colors = []
        
        if option == 'rgb':
            for s in source:
                self.colors.append(Color(s, is_RGB=True))
        elif option == 'hex':
            for s in source:
                self.colors.append(Color(s, is_RGB=False))
        elif option == 'color':
            for s in source:
                self.colors.append(copy.deepcopy(s))
        else:
            raise ValueError("option must be 'rbg', 'color', or 'hex'")

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