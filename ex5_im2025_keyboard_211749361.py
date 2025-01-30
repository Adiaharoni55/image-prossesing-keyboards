import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_edges(piece):
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

    kernel_size = (2, 2)  # You can adjust this size
    kernel = np.ones(kernel_size, np.uint8)
    
    # Apply morphological opening
    # piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, kernel)
    piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, kernel)
    return piece

def find_parallel_lines(matches, mac_kp, windows_kp, tolerance=0.11):
    if not matches:
        return []
    
    # Calculate and sort slopes
    slopes = []
    for m in matches:
        slope = calculate_slope(m, mac_kp, windows_kp)
        slopes.append((slope, m))
    
    # Sort by slope value
    slopes.sort(key=lambda x: x[0])
    
    groups = []
    used = set()
    
    for i, (slope1, match1) in enumerate(slopes):
        if i in used:
            continue
        
        current_group = [match1]
        used.add(i)
            
        # Check all remaining slopes
        for j in range(i + 1, len(slopes)):
            if j not in used:
                slope2, match2 = slopes[j]
                if abs(slope1 - slope2) <= tolerance:
                    current_group.append(match2)
                    used.add(j)
                else:
                    # Since slopes are sorted, if difference exceeds tolerance, 
                    # all remaining slopes will also exceed it
                    break
        
        groups.append(current_group)
    
    return max(groups, key=len) if groups else []

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

def calculate_distance(match1, match2, mac_kp, windows_kp):
    pt1_1 = mac_kp[match1.queryIdx].pt
    pt2_1 = windows_kp[match1.trainIdx].pt
    pt1_2 = mac_kp[match2.queryIdx].pt
    pt2_2 = windows_kp[match2.trainIdx].pt
    
    mid1 = ((pt1_1[0] + pt2_1[0])/2, (pt1_1[1] + pt2_1[1])/2)
    mid2 = ((pt1_2[0] + pt2_2[0])/2, (pt1_2[1] + pt2_2[1])/2)
    
    return ((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)**0.5

def calculate_slope(match, mac_kp, windows_kp):
    mac_pt = mac_kp[match.queryIdx].pt
    win_pt = windows_kp[match.trainIdx].pt
    return (win_pt[1] - mac_pt[1]) / (win_pt[0] - mac_pt[0]) if win_pt[0] != mac_pt[0] else float('inf')

def get_points(match, mac_kp, windows_kp):
    mac_pt = mac_kp[match.queryIdx].pt
    win_pt = windows_kp[match.trainIdx].pt
    return mac_pt, win_pt

def reduce_similar_matches(matches, mac_kp, windows_kp, slope_tolerance=0.04, dis_tolerance=0.05):
    if not matches:
        return []

    slopes = [(calculate_slope(m, mac_kp, windows_kp), m) for m in matches]
    reduced_matches = []
    used = set()

    for i, (slope1, match1) in enumerate(slopes):
        if i in used:
            continue

        used.add(i)
        reduced_matches.append(match1)

        mac_pt1, win_pt1 = get_points(match1, mac_kp, windows_kp)

        for j, (slope2, match2) in enumerate(slopes[i+1:], start=i+1):
            if j in used:
                continue

            mac_pt2, win_pt2 = get_points(match2, mac_kp, windows_kp)

            # # Check if either starting point is shared
            # if points_are_close(mac_pt1, mac_pt2) or points_are_close(win_pt1, win_pt2):
            #     if abs(slope1 - slope2) <= slope_tolerance:
            #         used.add(j)
            #         continue

            # Original distance check for non-shared points
            if abs(slope1 - slope2) <= slope_tolerance:
                distance = calculate_distance(match1, match2, mac_kp, windows_kp)
                if distance <= dis_tolerance:
                    used.add(j)

    return reduced_matches

def find_matching_pieces(piece_array_mac, piece_array_windows):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    all_matches = []
    
    for i, (mac_piece, mac_piece_contours) in enumerate(piece_array_mac):
        # if i < 43:
        #     continue
        h, w = mac_piece.shape  
        mac_relation_area_height = max(h/w, w/h)
        
        mac_matches = []
        mac_kp, mac_des = sift.detectAndCompute(mac_piece, None)
        
        if mac_des is None:
            continue
            
        for j, (windows_piece, win_piece_contours) in enumerate(piece_array_windows):
            h, w = windows_piece.shape  
            win_relation_area_height = max(h/w, w/h)

            if abs(mac_relation_area_height - win_relation_area_height) > 0.2:
                continue

            count_mac_contours = len(mac_piece_contours)
            count_win_contours = len(win_piece_contours)

            if(abs(count_win_contours - count_mac_contours)) != 0:
                continue

            windows_kp, windows_des = sift.detectAndCompute(windows_piece, None)
            
            if windows_des is None:
                continue
                
            matches = bf.knnMatch(mac_des, windows_des, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.875 * n.distance:
                    good_matches.append(m)

            reduced_matches = reduce_similar_matches(good_matches, mac_kp, windows_kp)
            parallel_matches = find_parallel_lines(reduced_matches, mac_kp, windows_kp)
            score = len(parallel_matches)

            if score > 2:
                match_info = {
                        'mac_piece': mac_piece,
                        'windows_piece': windows_piece,
                        'original_mac_piece': piece_array_mac[i],
                        'original_windows_piece': piece_array_windows[j],
                        'mac_index': i,
                        'windows_index': j,
                        'score': score,
                        'matches': parallel_matches,
                        'mac_kp': mac_kp,
                        'windows_kp': windows_kp,
                        'total_matches': len(reduced_matches),
                        'CA_big': 0,
                        'CA_small': 0,
                        'mac_contours': mac_piece_contours,
                        'windows_contours': win_piece_contours

                    }
            

                if (len(win_piece_contours) != 0 and len(mac_piece_contours) != 0):
                    win_check_biggest_contour = (win_piece_contours[0]["area"] / windows_piece.size)*100
                    mac_check_biggest_contour = (mac_piece_contours[0]["area"] / mac_piece.size)*100
                    CA_count = abs(win_check_biggest_contour - mac_check_biggest_contour)
                    match_info['CA_bigest'] = CA_count

                    win_check_biggest_contour = (win_piece_contours[-1]["area"] / windows_piece.size)*100
                    mac_check_biggest_contour = (mac_piece_contours[-1]["area"] / mac_piece.size)*100
                    CA_count = abs(win_check_biggest_contour - mac_check_biggest_contour)
                    match_info['CA_small'] = CA_count


                if score >= 5:
                    mac_matches.append(match_info)

                elif min(match_info['CA_bigest'], match_info['CA_small']) < 0.55 and max(match_info['CA_bigest'], match_info['CA_small']) < 2:
                    mac_matches.append(match_info)
                            

        if mac_matches:
            sorted_matches = sorted(mac_matches, 
                                 key=lambda x: (x['score'], x['total_matches']), 
                                 reverse=True)

            if sorted_matches[0]['score'] > 5:
                if len(sorted_matches) > 1 and sorted_matches[1]['score'] < sorted_matches[0]['score'] - 1:
                    top_matches = [sorted_matches[0]]
                else:
                    top_matches = [m for m in sorted_matches if m['score'] >= sorted_matches[0]['score'] - 1]

            else:
                top_matches = sorted_matches[:6]

            # visualize_matches_detailed(top_matches, windows_keyboard, bbox_coords)

            all_matches.append(top_matches)

    return all_matches
      
def visualize_matches_detailed(all_matches, windows_keyboard, bbox_coords):
    # Iterate through each set of matches
    for mac_matches in all_matches:
        n_matches = len(mac_matches)
        # Create a 2x8 subplot grid
        fig, axes = plt.subplots(2, min(8, n_matches), figsize=(20, 7))
        
        # Make sure axes is always 2D
        if n_matches == 1:
            axes = np.array([[axes[0]], [axes[1]]])  # Reshape for single match case
        
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
            axes[0, idx].set_title(f'score: {match["score"]}\n')
            axes[0, idx].axis('off')

            # Bottom row: highlight piece location in keyboard
            keyboard_viz = cv2.cvtColor(windows_keyboard.copy(), cv2.COLOR_GRAY2BGR)
            windows_idx = match['windows_index']
            x1, y1, x2, y2 = bbox_coords[windows_idx]
            
            # Draw rectangle around matched piece
            cv2.rectangle(keyboard_viz, (x1, y1), (x2, y2), (0, 255, 0), 15)
            
            axes[1, idx].imshow(keyboard_viz)
            axes[1, idx].axis('off')
        
        # Hide empty subplots if less than 8 matches
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
        plt.show()

def process_windows_keyboard(windows_binary):
    contours, _ = cv2.findContours(windows_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces_array_windows = []
    bbox_coords = {}
    
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 100:
            piece = windows_binary[y:y+h, x:x+w]
            pieces_array_windows.append(piece)
            bbox_coords[len(pieces_array_windows)-1] = (x, y, x+w, y+h)
            
    return pieces_array_windows, bbox_coords

def extract_pieces(binary_image, min_contour_area=100, is_mac=True):
    """Extract keyboard pieces from binary image."""
    contours, _ = cv2.findContours(binary_image, 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    pieces_array = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            piece = crop_edges(binary_image[y:y+h, x:x+w])
            
            piece_contours = process_piece_contours(piece, is_mac)
            if piece_contours:  # Only add if valid contours were found
                pieces_array.append((piece, piece_contours))
    
    return pieces_array

def process_piece_contours(piece, is_mac):
    """Process contours for an individual keyboard piece."""
    contours, _ = cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_for_piece = []
    
    size_divisor = 500 if is_mac else 370
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        
        # Check if contour meets size and position criteria
        if ((piece.size / size_divisor) < area < (piece.size - piece.size / 1.17) and 
            10 < cx < (piece.shape[1]-10) and 10 < cy < (piece.shape[0]-10)):
            contour = {
                'x': cx,
                'y': cy,
                'width': width,
                'height': height,
                'area': area,
                'angle': angle,
                'cnt': cnt
            }
            contours_for_piece.append(contour)
    
    return sorted(contours_for_piece, key=lambda x: x['area'], reverse=True)

def reduce_matches(matching_pieces):
    """Reduce matches by eliminating duplicate windows piece matches."""
    sorted_matching_pieces = sorted(matching_pieces, key=lambda x: len(x))
    found_windows_indices = set()
    reduced_matches = []

    for matches in sorted_matching_pieces:
        filtered_matches = [
            match for match in matches 
            if match['windows_index'] not in found_windows_indices
        ]
        
        if len(filtered_matches) == 1 and filtered_matches[0]['score'] > 4:
            found_windows_indices.add(filtered_matches[0]['windows_index'])
            reduced_matches.append(filtered_matches)
        elif filtered_matches:
            reduced_matches.append(filtered_matches)
            
    return reduced_matches


def main():
    # Load images
    mac = cv2.imread("mac.jpeg", cv2.IMREAD_GRAYSCALE)
    windows = cv2.imread("windows.jpeg", cv2.IMREAD_GRAYSCALE)

    # proccess mac 
    mac_binary = cv2.threshold(mac, 135, 255,cv2.THRESH_BINARY)[1]
    mac_binary = np.where(mac_binary == 0, 255, 0).astype(np.uint8)
    kernel_size = (3, 3)  
    kernel = np.ones(kernel_size, np.uint8)
    mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)
    mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)

    # proccess windows 
    windows = cv2.imread("windows.jpeg", cv2.IMREAD_GRAYSCALE)
    windows_binary = np.where(windows < 170, 255, 0).astype(np.uint8)

        # Extract pieces from both keyboards
    pieces_array_mac = extract_pieces(mac_binary, is_mac=True)
    pieces_array_windows = extract_pieces(windows_binary, is_mac=False)

    # Process windows keyboard for visualization
    _, bbox_coords = process_windows_keyboard(windows_binary)
    
    # Find and reduce matches
    print("Finding matches...")
    matching_pieces = find_matching_pieces(pieces_array_mac, pieces_array_windows)
    reduced_matches = reduce_matches(matching_pieces)
    
    # Visualize results
    visualize_matches_detailed(reduced_matches, windows, bbox_coords)

    # # create gif:
    # import make_animation
    # make_animation.create_matches_animation(reduced_matches, windows, bbox_coords)


if __name__ == "__main__":
    main()
