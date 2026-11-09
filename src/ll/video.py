"""Phase 7b: Video Processing, Annotation, Filmstrips, and Side-by-Side Compositing."""
from __future__ import annotations
import argparse
import pathlib
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_mp4(frames: list[np.ndarray], path: str | pathlib.Path, fps: int = 50):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps)


def write_gif(frames: list[np.ndarray], path: str | pathlib.Path, fps: int = 25, stride: int = 2, max_width: int = 480):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    subsampled = frames[::stride]
    resized = []
    for f in subsampled:
        img = Image.fromarray(f)
        if img.width > max_width:
            w_percent = max_width / float(img.width)
            h_size = int(float(img.height) * float(w_percent))
            img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)
        resized.append(np.array(img))
    iio.imwrite(path, resized, fps=fps, loop=0)


def read_frames(path: str | pathlib.Path, stride: int = 1) -> list[np.ndarray]:
    frames = iio.imread(path)
    return [frames[i] for i in range(0, len(frames), stride)]


def annotate_frame(frame: np.ndarray, text_lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 10
    for line in text_lines:
        draw.text((10, y), line, fill=(255, 255, 255))
        y += 18
    return np.array(img)


def create_filmstrip(frames: list[np.ndarray], n: int = 8, out_path: str | None = None) -> Image.Image:
    if len(frames) == 0:
        return Image.new("RGB", (800, 200), (0, 0, 0))

    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected = [frames[i] for i in indices]

    # Resize each frame
    thumb_w, thumb_h = 240, 160
    thumbs = []
    for idx, f in zip(indices, selected):
        img = Image.fromarray(f).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), f"Frame {idx}", fill=(255, 220, 50))
        thumbs.append(img)

    # 4 columns x 2 rows
    cols, rows = 4, 2
    grid_img = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 25))
    for i, t in enumerate(thumbs):
        c = i % cols
        r = i // cols
        grid_img.paste(t, (c * thumb_w, r * thumb_h))

    if out_path:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(out_path)
    return grid_img


def build_all(seed: int = 0, videos_dir: str | pathlib.Path = "videos"):
    """Build the Phase 7c outputs from the streamed rollout videos.

    Inputs (produced by ``python -m ll.rollouts`` and ``scripts/01_random_agent.py``):
      videos/raw_{config}_seed{seed}_ep0.mp4   one streamed clip per trained config
      videos/random_crash.mp4                 random-policy crash clip

    Outputs:
      videos/{config}_seed{seed}.gif            per-config GIF
      videos/{config}_seed{seed}_filmstrip.png  per-config 8-frame filmstrip
      videos/random_crash.gif                   random crash GIF (if the mp4 exists)
      videos/comparison.mp4                     side-by-side of all raw clips

    Missing inputs are skipped with a note, never an error, so the driver is
    safe to run before every config has a rollout.
    """
    videos_dir = pathlib.Path(videos_dir)
    raw_paths = sorted(videos_dir.glob(f"raw_*_seed{seed}_ep0.mp4"))

    if not raw_paths:
        print(f"No streamed rollout videos found in {videos_dir} (expected raw_*_seed{seed}_ep0.mp4).")
        print("Run 'python -m ll.rollouts' first to collect them.")
    for raw_path in raw_paths:
        config = raw_path.name[len("raw_"):].split(f"_seed{seed}_ep0.mp4")[0]
        frames = read_frames(raw_path)
        gif_path = videos_dir / f"{config}_seed{seed}.gif"
        print(f"Building {gif_path} ({len(frames)} frames)...")
        write_gif(frames, gif_path)
        filmstrip_path = videos_dir / f"{config}_seed{seed}_filmstrip.png"
        print(f"Building {filmstrip_path}...")
        create_filmstrip(frames, out_path=filmstrip_path)

    random_mp4 = videos_dir / "random_crash.mp4"
    if random_mp4.exists():
        print(f"Building {videos_dir / 'random_crash.gif'}...")
        write_gif(read_frames(random_mp4), videos_dir / "random_crash.gif")
    else:
        print(f"Skipping random_crash.gif: {random_mp4} not found (run scripts/01_random_agent.py).")

    if len(raw_paths) >= 2:
        labels = [
            p.name[len("raw_"):].split(f"_seed{seed}_ep0.mp4")[0] for p in raw_paths
        ]
        side_by_side_video(
            [str(p) for p in raw_paths],
            labels,
            str(videos_dir / "comparison.mp4"),
        )
    elif raw_paths:
        print("Skipping comparison.mp4: need at least 2 raw videos for a side-by-side.")


def main():
    ap = argparse.ArgumentParser(description="Build Phase 7c videos (GIFs, filmstrips, side-by-side) from streamed rollout clips.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--videos-dir", type=str, default="videos")
    a = ap.parse_args()
    build_all(seed=a.seed, videos_dir=a.videos_dir)


def side_by_side_video(
    video_paths: list[str],
    labels: list[str],
    out_path: str,
    fps: int = 50,
):
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    all_video_frames = [read_frames(p) for p in video_paths]
    max_len = max(len(f) for f in all_video_frames)

    # Pad each video by freezing its last frame
    padded = []
    for f_list in all_video_frames:
        if len(f_list) < max_len:
            last_frame = f_list[-1]
            pad_needed = max_len - len(f_list)
            f_list = list(f_list) + [last_frame] * pad_needed
        padded.append(f_list)

    composite_frames = []
    for step in range(max_len):
        cols = []
        for i, f_list in enumerate(padded):
            f = f_list[step]
            img = Image.fromarray(f)
            # Annotate label
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, 30], fill=(10, 15, 25))
            draw.text((12, 6), labels[i], fill=(240, 240, 255))
            cols.append(np.array(img))

        # Stack horizontally
        row = np.hstack(cols)
        composite_frames.append(row)

    iio.imwrite(out_path, composite_frames, fps=fps)
    print(f"Side-by-side comparison video saved to {out_path} ({len(composite_frames)} frames).")



if __name__ == "__main__":
    main()
