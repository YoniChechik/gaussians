from pathlib import Path

import cv2

from data_handling.frame_extractor import VideoFrameExtractor


def run(input_dir_path: str | Path):
    input_dir_path = Path(input_dir_path)

    extractor = VideoFrameExtractor(input_dir_path / "Frames.m4v")

    save_dir_path = input_dir_path / "algo_debug" / "jpg_color_frames"
    save_dir_path.mkdir(parents=True, exist_ok=True)

    color_ind = 0
    while True:
        bgr_frame = extractor.extract_bgr_frame(color_ind)
        if bgr_frame is None:
            print("FINISH")
            break

        cv2.imwrite(save_dir_path / f"{color_ind:05d}.jpg", bgr_frame)
        print(color_ind)
        color_ind += 1


if __name__ == "__main__":
    path_list = [
        r"C:\Users\Yoni\Desktop\scans\2024-04-20T15-21-02FullScan",
        r"C:\Users\Yoni\Desktop\scans\2024-04-20T15-53-28secscan",
        r"C:\Users\Yoni\Desktop\scans\2024-04-20T16-32-05twoQR2sides",
        r"C:\Users\Yoni\Desktop\scans\2024-04-20T16-13-05TwoQR1",
    ]
    for path in path_list:
        run(path)
