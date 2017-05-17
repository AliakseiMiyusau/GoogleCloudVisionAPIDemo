from subprocess import call
import os

FFMPEG_CONVERT = "ffmpeg -i {} -filter:v fps=fps=2 {}%d.bmp"


def extract_frames(video_path, frame_dir_path):
    command = FFMPEG_CONVERT.format(video_path, frame_dir_path)
    print(command)
    call(command, shell=True)


def main():
    video = "/home/aliaksei/Documents/faces/test.mp4"
    frame_dir = "/home/aliaksei/Documents/faces/frames/"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    extract_frames(video, frame_dir)


if __name__ == "__main__":
    main()
