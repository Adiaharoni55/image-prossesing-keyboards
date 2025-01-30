import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os



def create_matches_animation(all_matches, windows_keyboard, bbox_coords, output_filename='matches_animation.gif'):
    
    # Pre-convert windows keyboard once instead of every frame
    keyboard_viz_base = cv2.cvtColor(windows_keyboard.copy(), cv2.COLOR_GRAY2BGR)
    
    # Optimize figure creation
    plt.ioff()
    images_pil = []
    print(f"Starting to process {len(all_matches)} sets of matches...")
    
    for i, mac_matches in enumerate(all_matches):
        fig = plt.figure(figsize=(20, 7), dpi=80)  # Reduced DPI for faster rendering
        gs = fig.add_gridspec(2, min(8, len(mac_matches)))
        
        for idx, match in enumerate(mac_matches):
            if idx >= 8:
                break
                
            # Top row
            ax_top = fig.add_subplot(gs[0, idx])
            matches_img = cv2.drawMatches(
                match['mac_piece'], match['mac_kp'],
                match['windows_piece'], match['windows_kp'],
                match['matches'], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            ax_top.imshow(matches_img)
            ax_top.set_title(f'score: {match["score"]}\n')
            ax_top.axis('off')
            
            # Bottom row - use pre-converted keyboard
            ax_bottom = fig.add_subplot(gs[1, idx])
            keyboard_viz = keyboard_viz_base.copy()
            windows_idx = match['windows_index']
            x1, y1, x2, y2 = bbox_coords[windows_idx]
            cv2.rectangle(keyboard_viz, (x1, y1), (x2, y2), (0, 255, 0), 15)
            ax_bottom.imshow(keyboard_viz)
            ax_bottom.axis('off')
        
        plt.suptitle(
            f'Matches for Mac Keyboard Piece #{mac_matches[0]["mac_index"]} ({i+1}/{len(all_matches)})',
            fontsize=12, y=1.02  # Reduced padding
        )
        
        plt.tight_layout()
        
        # Faster image conversion
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        pil_img = Image.fromarray(buf).convert('RGB')  # Convert to RGB for smaller file size
        
        # Faster resizing with reduced quality
        new_size = (pil_img.size[0] // 2, pil_img.size[1] // 2)
        pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)  # BILINEAR is faster than LANCZOS
        
        images_pil.append(pil_img)
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_matches)} frames...")
    
    print(f"Creating GIF with {len(images_pil)} frames...")
    images_pil[0].save(
        output_filename,
        save_all=True,
        append_images=images_pil[1:],
        optimize=False,  # Disable optimization for faster saving
        duration=1000,  # Reduced to 1 second per frame
        loop=0
    )
    
    print(f"Animation saved as {output_filename}")
    print(f"GIF file size: {os.path.getsize(output_filename) / (1024 * 1024):.2f} MB")


