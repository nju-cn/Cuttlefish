import track.feature as feature
import time
def evaluate(boxes_list0, boxes_list1, x_times, y_times, confidence):
    frame_num = len(boxes_list0)
    frame_index = 0
    scores = 0
    for frame_index in range(frame_num):
        TP_FN = len(boxes_list0[frame_index])
        TP_FP = len(boxes_list1[frame_index])

        TP = 0
        for box1 in boxes_list1[frame_index]:
            for box0 in boxes_list0[frame_index]:
                if IOU(box1[0] * x_times, box1[1] * y_times, box1[2] * x_times, box1[3] * y_times, box0[0], box0[1], box0[2], box0[3]) >= confidence:
                    TP += 1
                    break
        if TP != 0:
            scores += 2/(TP_FN/TP + TP_FP/TP)

    average_score = scores / frame_num
    return average_score

def IOU(x1,y1,w1,h1,x2,y2,w2,h2):
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return 0
    cross_square = width * height
    union_square = w1 * h1 + w2 * h2 - cross_square
    return cross_square / union_square

def EvaluateVelocity(firstimage,lastimage,boxlist1,boxlist2,slot_time=0.5):
    firstframe=feature.MyFrame(firstimage)
    lastframe=feature.MyFrame(lastimage)
    firstframe.initial_object(boxlist1,True)
    lastframe.initial_object(boxlist2,False)
    avg_dis=feature.track(firstframe,lastframe)
    avg_v=avg_dis*slot_time
    return avg_v
