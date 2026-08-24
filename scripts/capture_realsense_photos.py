"""Preview and capture one RGB photo from the table and wrist RealSense cameras.

The defaults match the camera configuration used by ``README_REI.md``:

    python scripts/capture_realsense_photos.py

Press S to save a PNG from each camera and Q (or Esc) to quit.  Use ``--once``
to save one pair immediately without opening a preview window.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


WIDTH = 640
HEIGHT = 480
FPS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-cam-serial", default="244422300964")
    parser.add_argument("--wrist-cam-serial", default="252122300792")
    parser.add_argument(
        "--table-exposure",
        type=float,
        default=180.0,
        help="Manual table-camera exposure in 0.1 ms units (default: 180).",
    )
    parser.add_argument("--table-gain", type=float, default=16.0)
    parser.add_argument(
        "--wrist-exposure",
        type=float,
        default=100.0,
        help="Manual wrist-camera exposure in 0.1 ms units (default: 100).",
    )
    parser.add_argument("--wrist-gain", type=float, default=16.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/realsense_photos"),
        help="Directory for captured PNG files.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Frame pairs to discard after applying camera settings.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Save one pair after warmup and exit without a preview window.",
    )
    return parser


def find_color_sensor(rs, profile, name: str):
    for sensor in profile.get_device().query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        if "RGB" in sensor_name or "Color" in sensor_name:
            return sensor
    raise RuntimeError(f"{name} has no RGB color sensor")


def start_camera(rs, serial: str, name: str, exposure: float, gain: float):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    print(f"[RS] connecting {name} serial={serial}")
    profile = pipeline.start(config)
    try:
        sensor = find_color_sensor(rs, profile, name)
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(exposure))
        sensor.set_option(rs.option.gain, float(gain))
        actual_exposure = sensor.get_option(rs.option.exposure)
        actual_gain = sensor.get_option(rs.option.gain)
        print(
            f"[RS] {name}: exposure={actual_exposure:g} "
            f"({actual_exposure / 10.0:g} ms), gain={actual_gain:g}"
        )
    except BaseException:
        pipeline.stop()
        raise
    return pipeline


def read_rgb(pipeline) -> np.ndarray:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if color:
            return np.asanyarray(color.get_data()).copy()


def save_pair(
    table_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    table_path = output_dir / f"{timestamp}_table_cam.png"
    wrist_path = output_dir / f"{timestamp}_wrist_cam.png"
    Image.fromarray(table_rgb).save(table_path)
    Image.fromarray(wrist_rgb).save(wrist_path)
    print(f"saved {table_path}")
    print(f"saved {wrist_path}")
    return table_path, wrist_path


def validate_args(args: argparse.Namespace) -> None:
    if args.table_cam_serial == args.wrist_cam_serial:
        raise ValueError("table and wrist camera serial numbers must be different")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    for name in ("table_exposure", "wrist_exposure"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    for name in ("table_gain", "wrist_gain"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Run this script with the same environment used for rollout."
        ) from exc

    table_pipeline = None
    wrist_pipeline = None
    pygame_initialized = False
    try:
        table_pipeline = start_camera(
            rs,
            args.table_cam_serial,
            "table_cam",
            args.table_exposure,
            args.table_gain,
        )
        wrist_pipeline = start_camera(
            rs,
            args.wrist_cam_serial,
            "wrist_cam",
            args.wrist_exposure,
            args.wrist_gain,
        )

        print(f"warming up for {args.warmup_frames} frame pairs")
        for _ in range(args.warmup_frames):
            read_rgb(table_pipeline)
            read_rgb(wrist_pipeline)

        if args.once:
            save_pair(read_rgb(table_pipeline), read_rgb(wrist_pipeline), args.output_dir)
            return

        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "Pygame is required for preview mode. Install pygame or use --once."
            ) from exc

        pygame.display.init()
        pygame.font.init()
        pygame_initialized = True
        try:
            screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
        except pygame.error as exc:
            raise SystemExit(
                f"Could not open the preview window: {exc}. "
                "If this is a headless session, use --once."
            ) from exc
        pygame.display.set_caption("RealSense photo capture [S save | Q quit]")
        font = pygame.font.SysFont("sans", 24)
        print("preview ready: press S to save both photos; Q or Esc to quit")
        running = True
        while running:
            table_rgb = read_rgb(table_pipeline)
            wrist_rgb = read_rgb(wrist_pipeline)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_s:
                        save_pair(table_rgb, wrist_rgb, args.output_dir)

            table_surface = pygame.surfarray.make_surface(
                np.transpose(table_rgb, (1, 0, 2))
            )
            wrist_surface = pygame.surfarray.make_surface(
                np.transpose(wrist_rgb, (1, 0, 2))
            )
            screen.blit(table_surface, (0, 0))
            screen.blit(wrist_surface, (WIDTH, 0))
            screen.blit(font.render("table_cam", True, (0, 255, 0)), (12, 8))
            screen.blit(font.render("wrist_cam", True, (0, 255, 0)), (WIDTH + 12, 8))
            pygame.display.flip()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if wrist_pipeline is not None:
            wrist_pipeline.stop()
        if table_pipeline is not None:
            table_pipeline.stop()
        if pygame_initialized:
            pygame.quit()


if __name__ == "__main__":
    main()
