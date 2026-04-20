import subprocess

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUT_W, OUT_H = 1080, 1920
SMOOTH = 0.98
JITTER_THRESHOLD = 1.6  # jitter threshold in pixels


def create_face_tracked_vertical_video(
    ffmpeg_path: str,
    input_mp4: str,
    output_mp4: str,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    smooth: float = SMOOTH,
) -> None:
    # Loading Up the Model
    base_options = python.BaseOptions(model_asset_path="deprecated/blaze_face_short_range.tflite")
    visionrunningmode = vision.RunningMode
    options = vision.FaceDetectorOptions(base_options=base_options, running_mode=visionrunningmode.VIDEO)
    detector = vision.FaceDetector.create_from_options(options)

    # Opening the input_mp4 with cv2
    video = cv2.VideoCapture(input_mp4)

    # Loading a pipe to output to output.mp4
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0
    ff_cmd = [
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-i",
        input_mp4,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        output_mp4,
    ]
    proc = subprocess.Popen([ffmpeg_path, *ff_cmd], stdin=subprocess.PIPE)

    frame_index = 0
    aspect_ratio = out_w / out_h
    smooth_x = None
    with detector:
        if not video.isOpened():
            print("Couldn't Open the video File")
        while video.isOpened():
            # Reading the video and converting the color format to
            ret, frame = video.read()
            if not ret or frame is None:
                break
            h, w, _ = frame.shape
            crop_w = int(h * (aspect_ratio))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int((frame_index) * 1000 / fps)
            frame_index += 1

            # Detecting Faces
            detection_result = detector.detect_for_video(mp_image, timestamp)
            # Cropping the Video
            center_x = w // 2
            if detection_result.detections:
                # Get the Tracked Face's Bounding Box
                bbox = detection_result.detections[0].bounding_box
                # Get the Horizontal Middle/Center of the Bounding Box
                center_x = int(bbox.origin_x + (bbox.width / 2))

                # Smoothing Out the Video
                if smooth_x is None:
                    smooth_x = float(center_x)
                else:
                    if abs(center_x - smooth_x) > JITTER_THRESHOLD:
                        smooth_x = smooth_x + (center_x - smooth_x) * smooth
            # Calculating crop Boundaries
            if smooth_x is None:
                smooth_x = w // 2
            smooth_x = int(smooth_x)
            x_start = max(0, min(smooth_x - (crop_w // 2), w - crop_w))
            x_end = x_start + crop_w

            cropped_frame = frame[0:h, x_start:x_end]
            if cropped_frame is None:
                continue
            cropped_frame = cv2.resize(cropped_frame, (out_w, out_h))

            # Uncomment below to see preview of the Face Tracking
            # cv2.imshow('Vertical Face Follow',cropped_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

            try:
                proc.stdin.write(cropped_frame.tobytes())
            except BrokenPipeError:
                break

    video.release()
    cv2.destroyAllWindows()
    if proc.stdin:
        proc.stdin.close()
    proc.wait()
