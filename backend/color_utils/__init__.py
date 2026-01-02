import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_color_palettes(list1, list2, output_file='color_comparison.png'):
    """
    Plots two lists of hex colors side-by-side as squares.
    """
    n1 = len(list1)
    n2 = len(list2)
    n_max = max(n1, n2)

    # Adjust figure size based on the number of colors
    fig, ax = plt.subplots(figsize=(n_max * 1.5, 4))

    # Plot first list on top (y=1 to y=2)
    for i, color in enumerate(list1):
        rect = patches.Rectangle((i, 1), 0.8, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(i + 0.4, 1.9, color, ha='center', va='bottom', family='monospace')

    # Plot second list on bottom (y=0 to y=1)
    for i, color in enumerate(list2):
        rect = patches.Rectangle((i, 0), 0.8, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(i + 0.4, -0.1, color, ha='center', va='top', family='monospace')

    # Formatting the plot
    ax.set_xlim(-0.2, n_max)
    ax.set_ylim(-0.8, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")


def plot_grouped_color_palettes(src1, src2, sort1, sort2, output_file='grouped_comparison.png'):
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