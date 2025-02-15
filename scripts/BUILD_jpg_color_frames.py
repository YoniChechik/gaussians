from pathlib import Path

import cv2


def run(input_dir_path: str | Path):
    input_dir_path = Path(input_dir_path)
    video_path = input_dir_path / "Frames.m4v"

    # Open video with cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    save_dir_path = input_dir_path / "algo_debug" / "jpg_color_frames"
    save_dir_path.mkdir(parents=True, exist_ok=True)

    color_ind = 0
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("FINISH")
            break
        if (save_dir_path / f"{color_ind:05d}.jpg").exists():
            color_ind += 1
            print(f"{color_ind} already exists")
            continue
        cv2.imwrite(save_dir_path / f"{color_ind:05d}.jpg", bgr_frame)
        print(color_ind)
        color_ind += 1

    # Release the video capture object
    cap.release()


if __name__ == "__main__":
    path_list = [r"C:\Users\Yoni\Desktop\ns_data\data\Regular-scan-1-lab-conditions"]
    for path in path_list:
        run(path)
