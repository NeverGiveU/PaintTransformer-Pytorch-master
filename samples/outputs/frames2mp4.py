import cv2
import os 
import numpy as np 
from tqdm import tqdm 


if __name__ == "__main__":
    name = "bingbing"
    pths = ["./{}/".format(name)]
    files = os.listdir(pths[0])
    # sort
    files.sort(key=lambda fname:int(fname.split(".")[0]))

    # merge
    # 1. read in the 1st frame
    arr = cv2.imread(pths[0]+files[0])
    h, w = arr.shape[:2]
    size = (w*len(pths), h)

    # 2. initialize a director
    fps = 8
    vid_writer = cv2.VideoWriter("{}.mp4".format(name), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    # 3. merge frames
    for f in tqdm(files):
        frame = []
        for pth in pths:
            arr = cv2.imread(pth+f)
            frame.append(arr)
        frame = np.concatenate(frame, axis=1)
        vid_writer.write(frame)
    print("Finish processing ...")