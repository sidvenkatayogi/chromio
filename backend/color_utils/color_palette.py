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
    
    # sourced from evaluation_hsl.py
    def _hsl_to_hex(self, hsl_str: str) -> str:
        """
            Convert HSL string format "(H, S%, L%)" to hex color "#RRGGBB".
        """
        # Parse HSL values
        match = re.match(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)', hsl_str)
        if not match:
            return "#000000"
        
        h, s, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        # Convert to 0-1 range
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0
        
        # HSL to RGB conversion
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
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        # Convert to 0-255 range and format as hex
        r_int = int(round(r * 255))
        g_int = int(round(g * 255))
        b_int = int(round(b * 255))
        
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


    @staticmethod
    def merge_to_single_palette(palettes):
        merged_hex_list = []
        original_palette_group = []

        for index, palette in enumerate(palettes):
            merged_hex_list += palette.to_hex_list()
            original_palette_group += [index for i in range(palette.length())]

        return ColorPalette(source=merged_hex_list, option='hex', original_palette_group=original_palette_group)
    
    @staticmethod
    def hsl_list_to_hex_list(self, hsl_list: list[str]) -> list[str]:
        """
            Convert HSL string list to Hex string list
        """
        hex_list = [ self._hsl_to_hex(hsl_str) for hsl_str in hsl_list ]
        return hex_list

    def __str__(self):
        return ' '.join(self.to_hex_list())