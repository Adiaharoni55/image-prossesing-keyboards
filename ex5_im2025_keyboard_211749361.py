import cv2
import numpy as np
import matplotlib.pyplot as plt

animation = True

def filter_contained_contours(contours):
    # If no contours, return empty list
    if not contours:
        return []
        
    filtered_contours = contours.copy()
    filtered_contours.sort(key=lambda x: x['area'], reverse=True)  # Largest first
    
    filtered_results = [filtered_contours[0]]  # Keep the largest contour
    
    # Check each smaller contour against larger ones
    for small_piece in filtered_contours[1:]:
        is_contained = False
        
        # Get the center point of small piece
        center_point = (small_piece['x'], small_piece['y'])
        
        for larger_piece in filtered_results:
            # Check if the center point is inside the larger contour
            result = cv2.pointPolygonTest(larger_piece['cnt'], center_point, False)
            
            if result > 0:  # If center point is inside
                is_contained = True
                break
        
        if not is_contained:
            filtered_results.append(small_piece)
                
    return filtered_results


def crop_edges(piece, to_sharp=True):
    """
    Crop edges from a piece by checking what size is the black frame and cut it.
    The piece is assumed to be a binary/grayscale image where 0 represents black.
    
    Parameters:
        piece: numpy array of shape (height, width)
        
    Returns:
        Cropped numpy array with black frame removed
    """
    height, width = piece.shape
    
    # Get middle row and column for scanning
    mid_row = height // 2
    mid_col = width // 2
    
    # Scan from left
    crop_left = 0
    while crop_left < width and piece[mid_row, crop_left] == 0:
        crop_left += 1
    
    # Scan from right
    crop_right = width - 1
    while crop_right >= 0 and piece[mid_row, crop_right] == 0:
        crop_right -= 1
    
    # Scan from top
    crop_top = 0
    while crop_top < height and piece[crop_top, mid_col] == 0:
        crop_top += 1
    
    # Scan from bottom
    crop_bottom = height - 1
    while crop_bottom >= 0 and piece[crop_bottom, mid_col] == 0:
        crop_bottom -= 1
    
    # Return cropped image
    piece = piece[crop_top:crop_bottom+1, crop_left:crop_right+1]

    if to_sharp:
        kernel_size = (2, 2)  
        kernel = np.ones(kernel_size, np.uint8)
        piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, kernel)
    return piece


def draw_contours_on_piece(piece_img, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours on the input image
    
    Args:
        piece_img: Input image (make sure it's a color image)
        contours: List of contours from cv2.findContours
        color: BGR color tuple for contour lines (default: green)
        thickness: Thickness of contour lines (default: 2)
    
    Returns:
        Image with drawn contours
    """
    # Convert to color if image is grayscale
    if len(piece_img.shape) == 2:
        display_img = cv2.cvtColor(piece_img, cv2.COLOR_GRAY2BGR)
    else:
        display_img = piece_img.copy()
        
    # Draw all contours
    cv2.drawContours(display_img, contours, -1, color, thickness)
    
    return display_img


def visualize_matches_detailed(all_matches, windows_keyboard):
    # Iterate through each set of matches
    for mac_matches in all_matches:
        n_matches = len(mac_matches)
        fig, axes = plt.subplots(2, min(8, n_matches), figsize=(20, 7))
        
        # Make sure axes is always 2D
        if n_matches == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        
        for idx, match in enumerate(mac_matches):
            if idx >= 8:
                break

            # Draw matches between pieces
            matches_img = cv2.drawMatches(
                match['mac_piece'], match['mac_kp'],
                match['windows_piece'], match['windows_kp'],
                match['matches'], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Top row: show match details
            axes[0, idx].imshow(matches_img)
            axes[0, idx].set_title(f'score: {round(match["score"], 2)}\n')
            axes[0, idx].axis('off')

            # Bottom row: highlight piece location in keyboard
            keyboard_viz = cv2.cvtColor(windows_keyboard.copy(), cv2.COLOR_GRAY2BGR)
            
            # Get the original rectangle from the match data
            org_rect = match['original_windows_piece'][3]  # Get the org_rec from the tuple
            
            # Convert minAreaRect to points and draw
            box = cv2.boxPoints(org_rect)
            box = np.int0(box)
            cv2.drawContours(keyboard_viz, [box], 0, (0, 255, 0), 15)
            
            axes[1, idx].imshow(keyboard_viz)
            axes[1, idx].axis('off')
        
        # Hide empty subplots if less than 8 matches
        for j in range(n_matches, 8):
            if j < axes.shape[1]:
                axes[0, j].axis('off')
                axes[1, j].set_visible(False)
                
        plt.suptitle(
            f'Matches for Source Keyboard Piece #{mac_matches[0]["mac_index"]}\n\n' + 
            'Top Row: Feature Matching Visualization\n' +
            '(Source piece on left, Dest piece on right)\n\n' +
            'Bottom Row: Location in Dest Keyboard (green rectangle)',
            fontsize=12, y=1.05
        )
        plt.tight_layout()
        plt.show()


def process_windows_keyboard(windows_binary):
    # Apply some preprocessing to improve contour detection
    kernel = np.ones((3,3), np.uint8)
    windows_binary = cv2.morphologyEx(windows_binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours with different parameters
    contours, _ = cv2.findContours(
        windows_binary, 
        cv2.RETR_EXTERNAL,  # Try RETR_LIST or RETR_TREE if this doesn't work
        cv2.CHAIN_APPROX_TC89_KCOS  # More precise approximation
    )
    
    pieces_array_windows = []
    bbox_coords = {}
    
    min_area = 100  # Adjust this threshold if needed
    
    for idx, cnt in enumerate(contours):
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate traditional bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Add some padding to the bounding box
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = w + 2 * padding
            h = h + 2 * padding
            
            piece = windows_binary[y:y+h, x:x+w]
            pieces_array_windows.append(piece)
            bbox_coords[len(pieces_array_windows)-1] = (x, y, x+w, y+h)
    
    return pieces_array_windows, bbox_coords


def find_contours(binary_img, min_thresh=500, max_thresh=1.17, to_sharp=True):
    contours, _ = cv2.findContours(binary_img, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

    pieces_array = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        org_rec = cv2.minAreaRect(cnt)
        if cv2.contourArea(cnt) > 100:  # Filter small contours
            piece = crop_edges(binary_img[y:y+h, x:x+w], to_sharp)
            # Store original coordinates with current piece index
            mac_piece_contours, _ = cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_for_piece = []
            
            for cnt in mac_piece_contours:
                area = cv2.contourArea(cnt)
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rect
                if ((piece.size / min_thresh) < area < (piece.size - piece.size / max_thresh) and 
                    10 < cx < (piece.shape[1]-10) and 10 < cy < (piece.shape[0]-10)):
                    contour = {'x': cx, 'y': cy, 'width': width, 'height': height, 'area': area, 'angle': angle, 'cnt': cnt}
                    contours_for_piece.append(contour)
            
            # Include the original index with each piece
            pieces_array.append((piece, contours_for_piece, len(contours_for_piece), org_rec))

    # Process pieces while maintaining original indices
    for piece_idx, (piece, contours, length_org, orig_idx) in enumerate(pieces_array):
        filtered_contours = filter_contained_contours(contours)
        pieces_array[piece_idx] = (piece, filtered_contours, length_org, orig_idx)

    # Sort pieces but keep original index
    pieces_array = sorted(pieces_array, key=lambda x: x[2], reverse=True)
    
    return pieces_array


def visualize_contours(piece_array):
    for piece, contours, _, _ in piece_array:
        piece_color = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
        for contour_data in contours:
            # Correctly reconstruct the rotated rectangle
            rect = ((contour_data['x'], contour_data['y']), 
                    (contour_data['width'], contour_data['height']), 
                    contour_data['angle'])
            # Get the box points
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # Convert to integer coordinates
            
            # Draw the contour in green
            cv2.drawContours(piece_color, [box], 0, (0, 255, 0), 2)
                
        # Show each piece with contours
        plt.imshow(cv2.cvtColor(piece_color, cv2.COLOR_BGR2RGB))  # Convert to RGB for Matplotlib
        plt.title(f'{len(contours)}')
        plt.axis('off')
        plt.show()


def is_match_in_contour(match, contour, kp, is_query=True):
    """
    Check if a match point falls within a contour's bounds
    
    Args:
        match: DMatch object
        contour: Dictionary containing contour info
        kp: Keypoints list
        is_query: If True, use queryIdx, else use trainIdx
    """
    x = contour['x'] - contour['width']/2
    y = contour['y'] - contour['height']/2
    width = contour['width']
    height = contour['height']

    padding_x = width * 0.2
    padding_y = height * 0.2

    # Use queryIdx for Mac keypoints, trainIdx for Windows keypoints
    point = kp[match.queryIdx if is_query else match.trainIdx].pt

    return ((x - padding_x) <= point[0] <= (x + width + padding_x) and 
            (y - padding_y) <= point[1] <= (y + height + padding_y))


def match_contours(mac_piece_contours, win_piece_contours, good_matches, mac_kp, windows_kp):
    # Step 1: Create arrays of matches for each Mac contour
    matches_per_mac_contour = []
    for mac_contour in mac_piece_contours:
        # Get all matches that fall within this Mac contour
        contour_matches = [m for m in good_matches if is_match_in_contour(m, mac_contour, mac_kp, True)]
        matches_per_mac_contour.append(contour_matches)
    
    # Step 2: Sort Mac contours by number of matches (descending)
    sort_indices = np.argsort([len(matches) for matches in matches_per_mac_contour])[::-1]
    matches_per_mac_contour = [matches_per_mac_contour[i] for i in sort_indices]
    mac_piece_contours = [mac_piece_contours[i] for i in sort_indices]
    
    # Keep track of available Windows contours using a list of booleans
    available_win_indices = [True] * len(win_piece_contours)
    actual_matches = []  # Will store the final matches
    
    # Step 3 & 4: For each Mac contour (starting with most matches)
    for mac_matches in matches_per_mac_contour:
        if not mac_matches:
            continue
            
        # Find how many matches go to each available Windows contour
        best_match_count = 0
        best_matches = []
        best_win_idx = -1
        
        for win_idx, win_contour in enumerate(win_piece_contours):
            if not available_win_indices[win_idx]:
                continue
                
            # Get matches that fall within this Windows contour
            matches_in_win = [m for m in mac_matches 
                            if is_match_in_contour(m, win_contour, windows_kp, False)]
            
            if len(matches_in_win) > best_match_count:
                best_match_count = len(matches_in_win)
                best_matches = matches_in_win
                best_win_idx = win_idx
        
        if best_win_idx >= 0:
            # Add the matches between these contours to actual_matches
            actual_matches.extend(best_matches)
            # Mark the Windows contour as used
            available_win_indices[best_win_idx] = False
    
    return actual_matches


def find_matching_pieces(piece_array_mac, piece_array_windows, max_gap=2.5):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    all_matches = []
    
    for i, (mac_piece, mac_piece_contours, length_mac_contours_org, _) in enumerate(piece_array_mac):
        mac_matches = []
        mac_kp, mac_des = sift.detectAndCompute(mac_piece, None)  
        
        if mac_des is None: 
            continue
            
        for j, (windows_piece, win_piece_contours, length_win_contours_org, _) in enumerate(piece_array_windows):
            
            if(length_win_contours_org < length_mac_contours_org-1): 
                break
                
            if length_mac_contours_org != length_win_contours_org or len(mac_piece_contours) != len(win_piece_contours):
                continue

            windows_kp, windows_des = sift.detectAndCompute(windows_piece, None)  
            
            if windows_des is None: 
                continue
                
                        
            matches = bf.knnMatch(mac_des, windows_des, k=2)
            
            good_matches = []
            
            for match in matches:
                if len(match) < 2:
                    continue  # Skip if we don't have two matches to compare
                m, n = match  # Now we know we have 2 matches
                if m.distance < 0.875 * n.distance:
                    good_matches.append(m)
            
            actual_matches = match_contours(mac_piece_contours, win_piece_contours, good_matches, mac_kp, windows_kp)

            if len(mac_piece_contours) > 0:
                score = len(actual_matches) / len(mac_piece_contours)
            else:
                score = len(actual_matches)
                 
            if score >= 1.0:
                match_info = {
                        'mac_piece': mac_piece,
                        'windows_piece': windows_piece,
                        'original_mac_piece': piece_array_mac[i],
                        'original_windows_piece': piece_array_windows[j],
                        'mac_index': i,
                        'windows_index': j,
                        'score': score,
                        'matches': actual_matches,
                        'mac_kp': mac_kp,
                        'windows_kp': windows_kp,
                        'mac_contours': mac_piece_contours,
                        'windows_contours': win_piece_contours,

                    }
                mac_matches.append(match_info)

            
        if mac_matches:
            sorted_matches = sorted(mac_matches, 
                                key=lambda x: x['score'], 
                                reverse=True)
            
            top_matches = []
            top_score = sorted_matches[0]['score']
            unique_scores_found = 1  
            last_different_score = top_score
            max_matches = 8  
            
            for m in sorted_matches:
                current_score = m['score']
                
                # If we've already found 3 different scores or hit max matches, stop
                if unique_scores_found >= 3 or len(top_matches) >= max_matches:
                    break
                    
                # If it's the same score as we've seen, add it (subject to max_matches)
                if current_score == last_different_score:
                    if len(top_matches) < max_matches:
                        top_matches.append(m)
                # If it's a new score within acceptable gap, add it
                elif current_score >= (top_score - max_gap):
                    if len(top_matches) < max_matches:
                        top_matches.append(m)
                        unique_scores_found += 1
                        last_different_score = current_score
                else:
                    break  
                    
            all_matches.append(top_matches)  


    return sorted(all_matches, key=lambda x: (len(x), -x[0]['score']))


def reduce_matches(matching_pieces, max_thresh=4.5):
    """Reduce matches by eliminating duplicate windows piece matches."""
    found_windows_indices = set()
    reduced_matches = []

    for matches in matching_pieces:
        filtered_matches = [
            match for match in matches 
            if match['windows_index'] not in found_windows_indices
        ]
        
        if len(filtered_matches) == 1 and filtered_matches[0]['score'] > max_thresh:
            found_windows_indices.add(filtered_matches[0]['windows_index'])
            reduced_matches.append(filtered_matches)
        elif filtered_matches:
            reduced_matches.append(filtered_matches)            
    return reduced_matches


def main(source_binary, dest_binary, dest_org_img, max_gap_thresh = 2.5, animation_name=None):

    pieces_array_source  = find_contours(source_binary)

    pieces_array_dest = find_contours(dest_binary, 370)

    matching_pieces = find_matching_pieces(pieces_array_source, pieces_array_dest, max_gap_thresh)
    reduced_matches = reduce_matches(matching_pieces)
    visualize_matches_detailed(reduced_matches, dest_org_img)

    # *****************************************************************
    # to activate animation remove those lines (477-478) from comment:

    # import make_animation
    # make_animation.create_matches_gif(reduced_matches, dest_org_img, animation_name, duration=0.8)
   


if __name__ == '__main__':
    # ---------------------- PART 1 ----------------------
    # compering mac keyboard to windows keyboard 
    mac = cv2.imread("mac.jpeg", cv2.IMREAD_GRAYSCALE)

    mac_binary = cv2.threshold(mac, 135, 255,cv2.THRESH_BINARY)[1]

    mac_binary = np.where(mac_binary == 0, 255, 0).astype(np.uint8)
    kernel_size = (3, 3)  
    kernel = np.ones(kernel_size, np.uint8)
    mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)
    mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)

    windows = cv2.imread("windows.jpeg", cv2.IMREAD_GRAYSCALE)

    windows_binary = np.where(windows < 170, 255, 0).astype(np.uint8)

    cv2.imwrite('binary_windows.jpeg', windows_binary)

    main(mac_binary, windows_binary, windows, animation_name='keyboard_matches_win_vs_mac.gif')

    # ---------------------- PART 2 ----------------------
    # conpering pink mac keyboard to white mac keyboard

    pink_mac = cv2.imread("pink_mac.jpg", cv2.IMREAD_GRAYSCALE)

    pink_mac_binary = cv2.threshold(pink_mac, 100, 255,cv2.THRESH_BINARY)[1]

    pink_mac_binary = np.where(pink_mac_binary == 0, 255, 0).astype(np.uint8)
    kernel_size = (3, 3)  
    kernel = np.ones(kernel_size, np.uint8)
    pink_mac_binary = cv2.morphologyEx(pink_mac_binary, cv2.MORPH_CLOSE, kernel)
    pink_mac_binary = cv2.morphologyEx(pink_mac_binary, cv2.MORPH_CLOSE, kernel)

    # *****************************************************************
    # to activate pink_mac_binary vs white mac remove this line (519) from comment:
    
    # main(pink_mac_binary, mac_binary, mac, 1.0, animation_name='keyboard_matches_pinc_vs_white.gif')

