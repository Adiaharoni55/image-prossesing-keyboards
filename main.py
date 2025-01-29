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


def find_parallel_lines(matches, mac_kp, windows_kp, tolerance=0.1):
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


def points_are_close(pt1, pt2, tolerance=2):
    return abs(pt1[0] - pt2[0]) <= tolerance and abs(pt1[1] - pt2[1]) <= tolerance


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

            # Check if either starting point is shared
            if points_are_close(mac_pt1, mac_pt2) or points_are_close(win_pt1, win_pt2):
                if abs(slope1 - slope2) <= slope_tolerance:
                    used.add(j)
                    continue

            # Original distance check for non-shared points
            if abs(slope1 - slope2) <= slope_tolerance:
                distance = calculate_distance(match1, match2, mac_kp, windows_kp)
                if distance <= dis_tolerance:
                    used.add(j)

    return reduced_matches


def find_matching_pieces(piece_array_mac, piece_array_windows, windows_keyboard, bbox_coords):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    
    for i, mac_piece in enumerate(piece_array_mac):
        # if i < 55:
        #     continue
        h, w = mac_piece.shape 
        mac_relation = min(h/w, w/h) 
        mac_contours, hierarchy = cv2.findContours(mac_piece, 
                                    cv2.RETR_LIST,  # Changed to RETR_LIST to get all contours
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(mac_contours) > 6:
            kernel_size = (4, 4)  # You can adjust this size
            kernel = np.ones(kernel_size, np.uint8)
            mac_piece = cv2.morphologyEx(mac_piece, cv2.MORPH_CLOSE, kernel)

                
        mac_matches = []
        mac_kp, mac_des = sift.detectAndCompute(mac_piece, None)
        
        if mac_des is None:
            continue
            
        for j, windows_piece in enumerate(piece_array_windows):
            h, w = windows_piece.shape
            win_relation = min(h/w, w/h)
            # if abs(win_relation - mac_relation) > 0.03:
            #     continue

            win_contours, hierarchy = cv2.findContours(windows_piece, 
                    cv2.RETR_LIST,  # Changed to RETR_LIST to get all contours
                    cv2.CHAIN_APPROX_SIMPLE)
            
            # if abs(len(mac_contours) - len(win_contours)) > 4:
            #     continue

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
                    'total_matches': len(reduced_matches)
                }

                
                match_info['mac_contours'] = len(mac_contours)

                match_info['win_contours'] = len(win_contours)

                mac_matches.append(match_info)


        if mac_matches:
            sorted_matches = sorted(mac_matches, 
                                 key=lambda x: (x['score'], x['total_matches']), 
                                 reverse=True)
            if sorted_matches[0]['score'] >= 6:
                top_matches = [s for s in sorted_matches if s['score'] == sorted_matches[0]['score']]
            
            else:
                top_matches = sorted_matches[:8]

            visualize_matches_detailed(top_matches, windows_keyboard, bbox_coords)
            

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


def visualize_matches_detailed(mac_matches, windows_keyboard, bbox_coords):
    n_matches = len(mac_matches)
    fig, axes = plt.subplots(2, min(8, n_matches), figsize=(20, 5))
    
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
        axes[0, idx].set_title(f'score: {match["score"]}\nMC: {match['mac_contours']}\nWC: {match['win_contours']}')
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


    
    plt.suptitle(f'Top matches for Mac piece {mac_matches[0]["mac_index"]}')
    
    plt.tight_layout()
    plt.show()


# read amd make binary
mac = cv2.imread("mac.jpeg", cv2.IMREAD_GRAYSCALE)

mac_binary = cv2.threshold(mac, 135, 255,cv2.THRESH_BINARY)[1]

mac_binary = np.where(mac_binary == 0, 255, 0).astype(np.uint8)
kernel_size = (3, 3)  # You can adjust this size
kernel = np.ones(kernel_size, np.uint8)
mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)

kernel_size = (2, 2)  # You can adjust this size
kernel = np.ones(kernel_size, np.uint8)
mac_binary = cv2.morphologyEx(mac_binary, cv2.MORPH_CLOSE, kernel)



cv2.imwrite('binary_mac.jpeg', mac_binary)

# Find contours on the binary image directly
contours, hierarchy = cv2.findContours(mac_binary, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)


pieces_array_mac = []

# Get individual rectangles from contours
for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
   if cv2.contourArea(cnt) > 100:  # Filter small contours
       piece = mac_binary[y:y+h, x:x+w]
       pieces_array_mac.append(crop_edges(piece))


windows = cv2.imread("windows.jpeg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

enhanced_image = clahe.apply(windows)

windows_binary = np.where(windows < 170, 255, 0).astype(np.uint8)

cv2.imwrite('binary_windows.jpeg', windows_binary)

# Find contours on the binary image directly
contours, hierarchy = cv2.findContours(windows_binary, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)

pieces_array_windows = []

# Get individual rectangles from contours
for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
   if cv2.contourArea(cnt) > 100:  
       piece = windows_binary[y:y+h, x:x+w]
       pieces_array_windows.append(crop_edges(piece))

pieces_array_windows, bbox_coords = process_windows_keyboard(windows_binary)
matching_pieces = find_matching_pieces(pieces_array_mac, pieces_array_windows, windows, bbox_coords)



# matching_pieces = find_matching_pieces(pieces_array_mac, pieces_array_windows)
# visualize_matches(matching_pieces)




