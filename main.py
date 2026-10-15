# Adi Aharoni

import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        kernel = np.ones((2, 2), np.uint8)
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


def draw_match_figure(source_matches, dest_org_keyboard, source_org_img):
    """
    Build the three-row figure for one source piece:
    source piece location, feature matches, and destination piece locations.
    Used both for on-screen display and for GIF frames.
    """
    n_cols = min(8, len(source_matches))

    fig = plt.figure(figsize=(18, 8))
    gs = plt.GridSpec(3, 1)
    source_ax = plt.subplot(gs[0])

    # Grids for the match visualizations and destination keyboard locations
    match_axes = [plt.subplot2grid((3, n_cols), (1, i)) for i in range(n_cols)]
    dest_axes = [plt.subplot2grid((3, n_cols), (2, i)) for i in range(n_cols)]

    # First row: source piece location on the source keyboard
    source_keyboard_viz = cv2.cvtColor(source_org_img.copy(), cv2.COLOR_GRAY2BGR)
    source_rect = source_matches[0]['original_mac_piece'][3]
    source_box = cv2.boxPoints(source_rect).astype(int)
    cv2.drawContours(source_keyboard_viz, [source_box], 0, (0, 255, 0), 15)
    source_ax.imshow(source_keyboard_viz)
    source_ax.axis('off')

    for idx, match in enumerate(source_matches[:n_cols]):
        # Middle row: feature matches between the two pieces
        matches_img = cv2.drawMatches(
            match['mac_piece'], match['mac_kp'],
            match['windows_piece'], match['windows_kp'],
            match['matches'], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        match_axes[idx].imshow(matches_img)
        match_axes[idx].set_title(f'score: {round(match["score"], 2)}', pad=1)
        match_axes[idx].axis('off')

        # Bottom row: piece location on the destination keyboard
        keyboard_viz = cv2.cvtColor(dest_org_keyboard.copy(), cv2.COLOR_GRAY2BGR)
        dest_rect = match['original_windows_piece'][3]
        dest_box = cv2.boxPoints(dest_rect).astype(int)
        cv2.drawContours(keyboard_viz, [dest_box], 0, (0, 255, 0), 15)
        dest_axes[idx].imshow(keyboard_viz)
        dest_axes[idx].axis('off')

    plt.suptitle('Keyboard Key Matching Analysis', fontsize=14, y=0.98, fontweight='bold')
    plt.figtext(
        0.5,
        0.94,
        'First row: Source piece location. Second row: matches. Third row: Dest piece location.',
        fontsize=10,
        ha='center',
        va='center'
    )
    plt.subplots_adjust(bottom=0.05, top=0.90, hspace=0.2)

    return fig


def visualize_matches_detailed(all_matches, dest_org_keyboard, source_org_img):
    for source_matches in all_matches:
        draw_match_figure(source_matches, dest_org_keyboard, source_org_img)
        plt.show()


def find_contours(binary_img, min_thresh=500, max_thresh=1.17, to_sharp=True):
    contours, _ = cv2.findContours(binary_img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    pieces_array = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        org_rect = cv2.minAreaRect(cnt)
        if cv2.contourArea(cnt) > 100:  # Filter small contours
            piece = crop_edges(binary_img[y:y+h, x:x+w], to_sharp)
            inner_contours, _ = cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_for_piece = []

            for inner_cnt in inner_contours:
                area = cv2.contourArea(inner_cnt)
                rect = cv2.minAreaRect(inner_cnt)
                (cx, cy), (width, height), angle = rect
                # Keep mid-sized contours whose center is at least 10 px from the piece border
                if ((piece.size / min_thresh) < area < (piece.size - piece.size / max_thresh) and
                        10 < cx < (piece.shape[1]-10) and 10 < cy < (piece.shape[0]-10)):
                    contour = {'x': cx, 'y': cy, 'width': width, 'height': height,
                               'area': area, 'angle': angle, 'cnt': inner_cnt}
                    contours_for_piece.append(contour)

            # Keep the piece's rotated rectangle on the original keyboard for visualization
            pieces_array.append((piece, contours_for_piece, len(contours_for_piece), org_rect))

    # Remove nested contours inside each piece
    for piece_idx, (piece, contours, length_org, org_rect) in enumerate(pieces_array):
        filtered_contours = filter_contained_contours(contours)
        pieces_array[piece_idx] = (piece, filtered_contours, length_org, org_rect)

    # Sort pieces by their original number of contours (largest first)
    pieces_array = sorted(pieces_array, key=lambda x: x[2], reverse=True)

    return pieces_array


def visualize_contours(piece_array):
    for piece, contours, _, _ in piece_array:
        piece_color = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
        for contour_data in contours:
            # Reconstruct the rotated rectangle
            rect = ((contour_data['x'], contour_data['y']),
                    (contour_data['width'], contour_data['height']),
                    contour_data['angle'])
            box = cv2.boxPoints(rect).astype(int)

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

    # Use queryIdx for source keypoints, trainIdx for destination keypoints
    point = kp[match.queryIdx if is_query else match.trainIdx].pt

    return ((x - padding_x) <= point[0] <= (x + width + padding_x) and
            (y - padding_y) <= point[1] <= (y + height + padding_y))


def match_contours(mac_piece_contours, win_piece_contours, good_matches, mac_kp, windows_kp):
    # Step 1: Create arrays of matches for each source contour
    matches_per_mac_contour = []
    for mac_contour in mac_piece_contours:
        # Get all matches that fall within this source contour
        contour_matches = [m for m in good_matches if is_match_in_contour(m, mac_contour, mac_kp, True)]
        matches_per_mac_contour.append(contour_matches)

    # Step 2: Sort source contours by number of matches (descending)
    sort_indices = np.argsort([len(matches) for matches in matches_per_mac_contour])[::-1]
    matches_per_mac_contour = [matches_per_mac_contour[i] for i in sort_indices]

    # Keep track of available destination contours
    available_win_indices = [True] * len(win_piece_contours)
    actual_matches = []  # Will store the final matches

    # Steps 3 & 4: For each source contour (starting with most matches)
    for mac_matches in matches_per_mac_contour:
        if not mac_matches:
            continue

        # Find how many matches go to each available destination contour
        best_match_count = 0
        best_matches = []
        best_win_idx = -1

        for win_idx, win_contour in enumerate(win_piece_contours):
            if not available_win_indices[win_idx]:
                continue

            # Get matches that fall within this destination contour
            matches_in_win = [m for m in mac_matches
                              if is_match_in_contour(m, win_contour, windows_kp, False)]

            if len(matches_in_win) > best_match_count:
                best_match_count = len(matches_in_win)
                best_matches = matches_in_win
                best_win_idx = win_idx

        if best_win_idx >= 0:
            # Add the matches between these contours to actual_matches
            actual_matches.extend(best_matches)
            # Mark the destination contour as used
            available_win_indices[best_win_idx] = False

    return actual_matches


def find_matching_pieces(piece_array_mac, piece_array_windows, max_gap=2.5):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    all_matches = []

    # Compute SIFT features for every destination piece once
    windows_features = [sift.detectAndCompute(piece, None) for piece, _, _, _ in piece_array_windows]

    start_idx = 0
    for i, (mac_piece, mac_piece_contours, length_mac_contours_org, _) in enumerate(piece_array_mac):
        mac_matches = []
        mac_kp, mac_des = sift.detectAndCompute(mac_piece, None)

        if mac_des is None:
            continue

        # Find where destination pieces match the current source piece length
        for j in range(start_idx, len(piece_array_windows)):
            length_win_contours_org = piece_array_windows[j][2]
            if length_win_contours_org >= length_mac_contours_org:
                start_idx = j
                break

        # Find where destination pieces become too small
        end_idx = len(piece_array_windows)
        for j in range(start_idx, len(piece_array_windows)):
            length_win_contours_org = piece_array_windows[j][2]
            if length_win_contours_org < length_mac_contours_org:
                end_idx = j
                break

        # Process only the relevant destination pieces
        for j in range(start_idx, end_idx):
            windows_piece, win_piece_contours, _, _ = piece_array_windows[j]

            if len(mac_piece_contours) != len(win_piece_contours):
                continue

            windows_kp, windows_des = windows_features[j]

            if windows_des is None:
                continue

            matches = bf.knnMatch(mac_des, windows_des, k=2)

            # Lowe's ratio test
            good_matches = []
            for match in matches:
                if len(match) < 2:
                    continue
                m, n = match
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
            sorted_matches = sorted(mac_matches, key=lambda x: x['score'], reverse=True)

            # Keep up to 3 distinct scores within max_gap of the best score, max 5 matches
            top_matches = []
            top_score = sorted_matches[0]['score']
            unique_scores_found = 1
            last_different_score = top_score
            max_matches = 5

            for m in sorted_matches:
                current_score = m['score']

                if unique_scores_found >= 3 or len(top_matches) >= max_matches:
                    break

                if current_score == last_different_score:
                    top_matches.append(m)
                elif current_score >= (top_score - max_gap):
                    top_matches.append(m)
                    unique_scores_found += 1
                    last_different_score = current_score
                else:
                    break

            all_matches.append(top_matches)

    return sorted(all_matches, key=lambda x: (len(x), -x[0]['score']))


def reduce_matches(matching_pieces, max_thresh=4.5):
    """
    Remove duplicate destination matches. Once a destination piece is a source
    piece's only remaining match with a score above max_thresh, it is locked
    and removed from the candidates of later source pieces.
    """
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


def main(source_binary, dest_binary, dest_org_img, source_org_img, max_gap_thresh=2.5, gif_path=None):
    """
    Match the pieces of the source keyboard to the destination keyboard and show the results.
    If gif_path is given, the results are also saved as a GIF (requires Pillow).
    """
    pieces_array_source = find_contours(source_binary)
    pieces_array_dest = find_contours(dest_binary, 370)

    matching_pieces = find_matching_pieces(pieces_array_source, pieces_array_dest, max_gap_thresh)
    reduced_matches = reduce_matches(matching_pieces)

    if gif_path:
        from create_gif import create_matches_gif
        create_matches_gif(reduced_matches, dest_org_img, source_org_img, output_path=gif_path)

    visualize_matches_detailed(reduced_matches, dest_org_img, source_org_img)


def binarize_mac(img, threshold):
    """Invert-threshold a Mac keyboard image and clean it with two 3x3 closings."""
    binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    binary = np.where(binary == 0, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


if __name__ == '__main__':
    # ---------------------- PART 1 ----------------------
    # Comparing the Mac keyboard to the Windows keyboard
    mac = cv2.imread("data/mac.jpeg", cv2.IMREAD_GRAYSCALE)
    mac_binary = binarize_mac(mac, 135)

    windows = cv2.imread("data/windows.jpeg", cv2.IMREAD_GRAYSCALE)
    windows_binary = np.where(windows < 170, 255, 0).astype(np.uint8)

    main(mac_binary, windows_binary, windows, mac)
    # To also save the result as a GIF:
    # main(mac_binary, windows_binary, windows, mac, gif_path="images/keyboard_matches_win_vs_mac.gif")

    # ---------------------- PART 2 ----------------------
    # Comparing the pink Mac keyboard to the white Mac keyboard
    pink_mac = cv2.imread("data/pink_mac.jpg", cv2.IMREAD_GRAYSCALE)
    pink_mac_binary = binarize_mac(pink_mac, 100)

    # To run this comparison, uncomment the line below:
    # main(pink_mac_binary, mac_binary, mac, pink_mac, max_gap_thresh=1.0)
    # To also save the result as a GIF:
    # main(pink_mac_binary, mac_binary, mac, pink_mac, max_gap_thresh=1.0,
    #      gif_path="images/keyboard_matches_pinc_vs_white.gif")
