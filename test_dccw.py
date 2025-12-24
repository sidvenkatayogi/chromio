from color_utils.color_palette import ColorPalette
from color_utils.dccw_measurer import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_grouped_palettes(src1, src2, sort1, sort2, output_file='dccw_sorted.png'):
    """
    Plots 4 palettes with a significant gap between Source and Sorted groups.
    """
    palettes = [src1, src2, sort1, sort2]
    n_max = max(len(p) for p in palettes)
    
    # Increase vertical figsize to accommodate the extra gap
    fig, ax = plt.subplots(figsize=(n_max * 1.5, 10))

    # Updated Y-offsets: 
    # Notice the jump from 1.5 to 3.5 creates the group gap
    y_offsets = [4.5, 3.5, 1.0, 0.0]
    
    for row_idx, palette in enumerate(palettes):
        y = y_offsets[row_idx]
        for i, color in enumerate(palette):
            rect = patches.Rectangle((i, y), 0.8, 0.8, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(i + 0.4, y + 0.82, color, ha='center', va='bottom', 
                    fontsize=9, family='monospace')

    # --- Labels placed relative to the new offsets ---
    
    # Placed above y=4.5
    ax.text(0, 5.5, 'SOURCE', ha='left', va='bottom', 
            fontsize=16, fontweight='bold', color='#333333')
    
    # Placed above y=1.0
    ax.text(0, 2.0, 'SORTED', ha='left', va='bottom', 
            fontsize=16, fontweight='bold', color='#333333')

    # Adjust limits to account for the larger coordinate span
    ax.set_xlim(-0.2, n_max) 
    ax.set_ylim(-0.8, 6.2) # Raised from 4.8 to 6.2 to fit the taller stack
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")


if __name__ == "__main__":
    # sample palette from https://github.com/SuziKim/DCCW-dataset/tree/main/LHSP/LHSP-k10-jitter0-replacement0
    source_hex_list = ["#1A1816", "#CB3217", "#CDA775", "#675B85", "#E9DDCB", "#987052", "#44447E", "#C39572", "#803223", "#C7655D"]
    target_hex_list = ["#655429", "#3D211E", "#33421B", "#5A506B", "#C9C7B0", "#323234", "#A6986B", "#5F2927", "#8F8973", "#93695B"]

    dccw_measurer = DccwMeasurer(
        source=source_hex_list,
        source_option='hex',
        target=target_hex_list,
        target_option='hex'
    )

    dccw_score_no_cycle = dccw_measurer.measure_dccw(reflect_cycle=False)
    dccw_score_with_cycle = dccw_measurer.measure_dccw(reflect_cycle=True)
    
    # Both the Diversity metric and DCCW metric are important for meaasuring the quality of the generation.
    # Low DCCW score typically means a tighter match against the target palette, however a LOW DCCW score along with low
    # diversity typically indicate a palette that is monotonic but contains a target color. This senerio should be avoided unless
    # the target palette also lacks diversity.
    # A high Diversity score typically means a more varied color palette.
    # A Diversity value similar to the target palette combined with a low DCCW core indicate that the model is able to
    # generate matching palettes with a broader range of colors rather than only recommending similar hues.
    print("-------------------------------")
    print("Source Palette Diversity: ", dccw_measurer.calculate_source_diversity())
    print("Target Palette Diversity: ", dccw_measurer.calculate_target_diversity())
    print("-------------------------------")
    print("DCCW score (no cycle): ", dccw_score_no_cycle)
    print("DCCW score (with cycle): ", dccw_score_with_cycle)
    print("-------------------------------")
    
    source_palette = dccw_measurer.get_source_HEX_before_sort()
    target_palette = dccw_measurer.get_target_HEX_before_sort()
    source_palette_after_sort = dccw_measurer.get_source_HEX_after_sort()
    target_palette_after_sort = dccw_measurer.get_target_HEX_after_sort()
    
    plot_grouped_palettes(source_palette, target_palette, source_palette_after_sort, target_palette_after_sort)
