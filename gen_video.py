from Tracker_v2 import *
from moviepy.editor import VideoFileClip
from time import time
import argparse

parser = argparse.ArgumentParser(description='process video file')
parser.add_argument('file', help='video file')
parser.add_argument('output', help='output video file')
parser.add_argument('-s','--start', help='video start time', type=float)
parser.add_argument('-e','--end', help='video end time', type=float)
args = parser.parse_args()

fname= args.file
input = f'test_videos/{fname}.mp4'
output = f'output_videos/{args.output}.mp4'


start = time()
tracker = Tracker()
input_clip = VideoFileClip(input).subclip(args.start, args.end) if args.start else VideoFileClip(input)
output_clip = input_clip.fl_image(tracker.pipeline)
output_clip.write_videofile(output, audio=False)
t = time()-start
print('video processing is completed in {} mins {} seconds'.format(int(t/60), round(t%60,2)))
