from pathlib import Path

import cv2

from data_handling.frame_extractor import VideoFrameExtractor


def m4v_to_mkv(input_filename: Path, output_filename: Path, output_fps: int = 30):
    extractor = VideoFrameExtractor(input_filename)

    cnt = 0
    is_first_frame = True

    while True:
        bgr_frame = extractor.extract_bgr_frame(cnt)
        if bgr_frame is None:
            break
        print(cnt)
        cnt += 1

        if is_first_frame:
            height, width = bgr_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_cap = cv2.VideoWriter(output_filename, fourcc, output_fps, (width, height))
            is_first_frame = False

        # Write the frame into the file 'output_filename'
        output_cap.write(bgr_frame)

    # Release everything when the job is finished
    output_cap.release()


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Yoni\Desktop\new_format_scans\2024-03-09T12-28-22-quick-markers-with"

    DATA_DIR = Path(DATA_DIR)

    input_fp = DATA_DIR / "Frames.m4v"

    out_dir = DATA_DIR / "algo_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fp = out_dir / "Frames.mkv"

    m4v_to_mkv(input_fp, out_fp)
