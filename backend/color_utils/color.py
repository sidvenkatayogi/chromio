from colormath.color_conversions import *
from colormath.color_objects import *

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)

class Color:
    def __init__(self, source, is_RGB = True):
        
        if is_RGB:
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
    
    def RGB_val(self):
        return self.rgb.get_upscaled_value_tuple()
    
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
        

    
