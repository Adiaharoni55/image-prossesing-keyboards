import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_matches_gif(all_matches, windows_keyboard, output_path='matches.gif', duration=1.0):
    """
    Creates a GIF using the same visualization style as visualize_matches_detailed
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
    
    for mac_matches in all_matches:
        n_matches = len(mac_matches)
        fig, axes = plt.subplots(2, min(8, n_matches), figsize=(20, 7), dpi=100)
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
    
    for i, mac_matches in enumerate(all_matches):
        n_matches = len(mac_matches)
        fig, axes = plt.subplots(2, min(8, n_matches), figsize=(20, 7), dpi=100)
        
        if n_matches == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        for idx, match in enumerate(mac_matches):
            if idx >= 8:
                break

            matches_img = cv2.drawMatches(
                match['mac_piece'], match['mac_kp'],
                match['windows_piece'], match['windows_kp'],
                match['matches'], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            axes[0, idx].imshow(matches_img)
            axes[0, idx].set_title(f'score: {round(match["score"], 2)}\n')
            axes[0, idx].axis('off')

            keyboard_viz = cv2.cvtColor(windows_keyboard.copy(), cv2.COLOR_GRAY2BGR)
            org_rect = match['original_windows_piece'][3]
            box = cv2.boxPoints(org_rect)
            box = np.int0(box)
            cv2.drawContours(keyboard_viz, [box], 0, (0, 255, 0), 15)
            
            axes[1, idx].imshow(keyboard_viz)
            axes[1, idx].axis('off')
        
        for j in range(n_matches, 8):
            if j < axes.shape[1]:
                axes[0, j].axis('off')
                axes[1, j].set_visible(False)
        
        plt.suptitle(
            f'Matches for Mac Keyboard Piece #{mac_matches[0]["mac_index"]}\n\n' + 
            'Top Row: Feature Matching Visualization\n' +
            '(Mac piece on left, Windows piece on right)\n\n' +
            'Bottom Row: Location in Windows Keyboard (green rectangle)',
            fontsize=12, y=1.05
        )
        plt.tight_layout()
        
        # Save figure to a temporary file
        temp_filename = f'temp_frame_{i}.png'
        plt.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Read the saved image with PIL
        img = Image.open(temp_filename)
        
        # Resize to match maximum dimensions
        if img.size[0] != max_width or img.size[1] != max_height:
            # Create a new white background image of max size
            new_img = Image.new('RGB', (max_width, max_height), 'white')
            # Calculate position to paste (center the smaller image)
            paste_x = (max_width - img.size[0]) // 2
            paste_y = (max_height - img.size[1]) // 2
            # Paste the original image onto the white background
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        
        # Optional: resize for reasonable file size, but maintain aspect ratio
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

