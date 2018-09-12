import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

POS_MSEC = cv2.CAP_PROP_POS_MSEC
POS_FRAMES = cv2.CAP_PROP_POS_FRAMES

def draw_text(frame, pt, frame_txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    sz, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
    baseline += thickness
    lower_left = (pt[0], pt[1])
    pt = (pt[0], pt[1]-baseline)
    upper_right = (pt[0]+sz[0], pt[1]-sz[1]-2)
    cv2.rectangle(frame, lower_left, upper_right, (0,0,0), -1, cv2.LINE_AA)
    cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness, cv2.LINE_AA)
    return lower_left, upper_right

def crossline(curr, prev, length):
    diff = curr - prev
    if diff[1] == 0:
        p1 = (int(curr[1]), int(curr[0]-length/2))
        p2 = (int(curr[1]), int(curr[0]+length/2))
    else:
        slope = -diff[0]/diff[1]
        x = np.cos(np.arctan(slope)) * length / 2
        y = slope * x
        p1 = (int(curr[1]-y), int(curr[0]-x))
        p2 = (int(curr[1]+y), int(curr[0]+x))
    return p1, p2

def get_props(captured):
    width = captured.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = captured.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

def get_video(videopath):
    return cv2.VideoCapture(videopath)

def get_frame(videopath, frame_id):
    captured = get_video(videopath)
    # x is the last item from the next_batch session above (461st trajectory)
    # it has [traj_no [step_no, [frame_id, x, y]]] position
    captured.set(POS_FRAMES, int(frame_id))

    frame_num = int(captured.get(POS_FRAMES))
    now = int(captured.get(POS_MSEC) / 1000)

    _, frame = captured.read()

    frame_txt = "{0}:{1}".format(now//60, now%60)
    frame_txt += ' (' + str(frame_num) + ')'

    return captured, frame, frame_txt

def plot(real_traj, complete_traj, statistics, frame_id, videopath):

    captured, frame, frame_txt = get_frame(videopath, frame_id)

    # agent_txt = 'Agent: {}'.format(int())
    pt = (3, frame.shape[0]-3)
    ll, ur = draw_text(frame, pt, frame_txt)

    # Plot the real trajectory
    color = (0, 255, 0)
    truth = []
    for idx, point in enumerate(real_traj[0, :, 1:]):
        point[0] = point[0]*(statistics[0][1] - statistics[0][0]) + statistics[0][0]
        point[1] = point[1]*(statistics[1][1] - statistics[1][0]) + statistics[1][0]
        truth.append(point)
    truth = np.array(truth)
    prev = truth[0]
    for curr in truth[1:]:
        loc1 = (int(prev[1]), int(prev[0])) # (y, x)
        loc2 = (int(curr[1]), int(curr[0])) # (y, x)
        p1, p2 = crossline(curr, prev, 3)
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        cv2.line(frame, loc1, loc2, color, 1, cv2.LINE_AA)
        prev = curr
    
    # Plot the predicted trajectory
    color = (192,19,192)
    pred = []
    for idx, point in enumerate(complete_traj):
        point[0] = point[0]*(statistics[0][1] - statistics[0][0]) + statistics[0][0]
        point[1] = point[1]*(statistics[1][1] - statistics[1][0]) + statistics[1][0]
        pred.append(point)
    pred = np.array(pred)
    prev = pred[0]
    for curr in pred[1:]:
        loc1 = (int(prev[1]), int(prev[0])) # (y, x)
        loc2 = (int(curr[1]), int(curr[0])) # (y, x)
        p1, p2 = crossline(curr, prev, 3)
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        cv2.line(frame, loc1, loc2, color, 1, cv2.LINE_AA)
        prev = curr

    plt.figure(figsize=(4, 4), dpi=960)
    plt.imshow(frame)
    plt.axis('off')
    plt.title("Plotting prediction in purple and targets in green")
    plt.show()