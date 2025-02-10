import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_matches_gif(all_matches, dest_org_keyboard, source_org_img, output_path='matches.gif', duration=0.8):
    """
    Creates a GIF using the same visualization style as visualize_matches_detailed
    Shows three rows: source keyboard piece location, match details, and destination locations
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install Pillow: pip install Pillow")
        return
        
    plt.ioff()  # Turn off interactive mode
    frames = []
    
    # First pass: find the maximum dimensions
    max_width = 0
    max_height = 0
    
    for source_matches in all_matches:
        n_matches = len(source_matches)
        # Create figure with same layout as visualization
        fig = plt.figure(figsize=(18, 8))
        gs = plt.GridSpec(3, 1)
        
        plt.tight_layout()
        
        # Save figure to get its size
        temp_filename = 'temp_size_check.png'
        plt.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        img = Image.open(temp_filename)
        max_width = max(max_width, img.size[0])
        max_height = max(max_height, img.size[1])
        os.remove(temp_filename)
    
    print(f"Starting to process {len(all_matches)} sets of matches...")
    print(f"Maximum dimensions: {max_width}x{max_height}")
    
    for i, source_matches in enumerate(all_matches):
        n_matches = len(source_matches)
        # Create figure with same layout as visualization
        fig = plt.figure(figsize=(18, 8))
        gs = plt.GridSpec(3, 1)
        
        # First row: Mac keyboard with piece location
        mac_ax = plt.subplot(gs[0])
        mac_keyboard_viz = cv2.cvtColor(source_org_img.copy(), cv2.COLOR_GRAY2BGR)
        mac_rect = source_matches[0]['original_mac_piece'][3]
        mac_box = np.int0(cv2.boxPoints(mac_rect))
        cv2.drawContours(mac_keyboard_viz, [mac_box], 0, (0, 255, 0), 15)
        mac_ax.imshow(mac_keyboard_viz)
        mac_ax.axis('off')
        
        # Create grids for matches and Windows keyboard locations
        match_axes = [plt.subplot2grid((3, min(8, n_matches)), (1, i)) for i in range(min(8, n_matches))]
        windows_axes = [plt.subplot2grid((3, min(8, n_matches)), (2, i)) for i in range(min(8, n_matches))]
        
        for idx, match in enumerate(source_matches):
            if idx >= 8:
                break

            # Draw matches between pieces
            matches_img = cv2.drawMatches(
                match['mac_piece'], match['mac_kp'],
                match['windows_piece'], match['windows_kp'],
                match['matches'], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Middle row: show match details
            match_axes[idx].imshow(matches_img)
            match_axes[idx].set_title(f'score: {round(match["score"], 2)}', pad=1)
            match_axes[idx].axis('off')

            # Bottom row: highlight piece location in Windows keyboard
            keyboard_viz = cv2.cvtColor(dest_org_keyboard.copy(), cv2.COLOR_GRAY2BGR)
            org_rect = match['original_windows_piece'][3]
            box = np.int0(cv2.boxPoints(org_rect))
            cv2.drawContours(keyboard_viz, [box], 0, (0, 255, 0), 15)
            windows_axes[idx].imshow(keyboard_viz)
            windows_axes[idx].axis('off')

        # Hide empty subplots
        for j in range(n_matches, 8):
            if j < len(match_axes):
                match_axes[j].axis('off')
                match_axes[j].set_visible(False)
                windows_axes[j].axis('off')
                windows_axes[j].set_visible(False)
        
        plt.suptitle(
            'Keyboard Key Matching Analysis',
            fontsize=14,
            y=0.98,
            fontweight='bold'
        )

        # Add subtitle with description
        plt.figtext(
            0.5,    
            0.94,  
            'First row: Source piece location. Second row: matches. Third row: Dest piece location.',
            fontsize=10, 
            ha='center',
            va='center'
        )

        # Adjust the subplot parameters
        plt.subplots_adjust(
            bottom=0.05, 
            top=0.90,   
            hspace=0.2    
        )
        
        # Save figure to a temporary file
        temp_filename = f'temp_frame_{i}.png'
        plt.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Read the saved image with PIL
        img = Image.open(temp_filename)
        
        # Resize to match maximum dimensions
        if img.size[0] != max_width or img.size[1] != max_height:
            new_img = Image.new('RGB', (max_width, max_height), 'white')
            paste_x = (max_width - img.size[0]) // 2
            paste_y = (max_height - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        
        # Optional: resize for reasonable file size
        scale_factor = 0.5  # Adjust this value as needed
        new_size = (int(max_width * scale_factor), int(max_height * scale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        frames.append(img)
        
        # Remove temporary file
        os.remove(temp_filename)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_matches)} frames...")
    
    if frames:
        print(f"\nCreating GIF with {len(frames)} frames...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=int(duration * 1000),
            loop=0
        )
        print(f"GIF saved to {output_path}")
        print(f"GIF file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    else:
        print("No frames were successfully processed")




