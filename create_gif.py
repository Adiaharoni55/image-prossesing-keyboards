import os

import matplotlib
import matplotlib.pyplot as plt

from main import draw_match_figure


def create_matches_gif(all_matches, dest_org_keyboard, source_org_img, output_path='matches.gif',
                       duration=0.8, scale_factor=0.5):
    """
    Create a GIF with one frame per source piece, using the same three-row layout
    as visualize_matches_detailed: source piece location, feature matches,
    and destination piece locations.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install Pillow: pip install Pillow")
        return

    if not all_matches:
        print("No matches to save")
        return

    was_interactive = matplotlib.is_interactive()
    plt.ioff()

    # Render every frame first
    raw_frames = []
    print(f"Rendering {len(all_matches)} frames...")
    for frame_idx, source_matches in enumerate(all_matches):
        fig = draw_match_figure(source_matches, dest_org_keyboard, source_org_img)
        temp_filename = f'temp_frame_{frame_idx}.png'
        fig.savefig(temp_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        with Image.open(temp_filename) as img:
            raw_frames.append(img.convert('RGB'))
        os.remove(temp_filename)

        if (frame_idx + 1) % 10 == 0:
            print(f"Rendered {frame_idx + 1}/{len(all_matches)} frames...")

    # Pad all frames to the size of the largest real frame, so nothing gets cropped
    max_width = max(img.size[0] for img in raw_frames)
    max_height = max(img.size[1] for img in raw_frames)
    new_size = (int(max_width * scale_factor), int(max_height * scale_factor))

    frames = []
    for img in raw_frames:
        canvas = Image.new('RGB', (max_width, max_height), 'white')
        canvas.paste(img, ((max_width - img.size[0]) // 2, (max_height - img.size[1]) // 2))
        frames.append(canvas.resize(new_size, Image.Resampling.LANCZOS))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=int(duration * 1000),
        loop=0
    )
    print(f"GIF saved to {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")

    if was_interactive:
        plt.ion()
