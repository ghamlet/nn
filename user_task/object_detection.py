import datetime
from ultralytics import YOLO
import cv2

from deep_sort_realtime.deepsort_tracker import DeepSort
CONFIDENCE_THRESHOLD = 0.1
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture("user_task/videos/6c3e546c-86d1-4fa1-afd8-a75ffc401ba2.mp4")
# initialize the video writer object


# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    frame = frame[::, 400:1400]
    # scale_percent = 0 # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
 

    #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])


    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
 
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()

cv2.destroyAllWindows()





# ####################
# CONFIDENCE_THRESHOLD = 0.2
# GREEN = (0, 255, 0)

# # initialize the video capture object
# video_cap = cv2.VideoCapture("user_task/videos/6c3e546c-86d1-4fa1-afd8-a75ffc401ba2.mp4")
# # initialize the video writer object


# # load the pre-trained YOLOv8n model
# model = YOLO("yolov8n.pt")
# while True:
#     # start time to compute the fps
#     start = datetime.datetime.now()

#     ret, frame = video_cap.read()

#     # if there are no more frames to process, break out of the loop
#     if not ret:
#         break

#     # run the YOLO model on the frame
#     detections = model(frame)[0]

#     for data in detections.boxes.data.tolist():
#         # extract the confidence (i.e., probability) associated with the detection
#         confidence = data[4]

#         # filter out weak detections by ensuring the 
#         # confidence is greater than the minimum confidence
#         if float(confidence) < CONFIDENCE_THRESHOLD:
#             continue

#         # if the confidence is greater than the minimum confidence,
#         # draw the bounding box on the frame
#         xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
#         cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

#     end = datetime.datetime.now()
#     # show the time it took to process 1 frame
#     total = (end - start).total_seconds()
#     print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

#     # calculate the frame per second and draw it on the frame
#     fps = f"FPS: {1 / total:.2f}"
#     cv2.putText(frame, fps, (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

#     # show the frame to our screen
#     cv2.imshow("Frame", frame)
  
#     if cv2.waitKey(1) == ord("q"):
#         break

# video_cap.release()

# cv2.destroyAllWindows()

#############################################
# conf_threshold = 0.4

# # Initialize the video capture and the video writer objects
# video_cap = cv2.VideoCapture("user_task/videos/0cdaf025-6350-4b8d-9da8-677c3a16c439.mp4")

# # Initialize the YOLOv8 model using the default weights
# model = YOLO("yolov8s.pt")

# while True:
#     # starter time to computer the fps
#     #start = datetime.datetime.now()
#     ret, frame = video_cap.read()
#     frame = frame[200:900, 400:1400]
#     #frame = frame[::2, ::2,::]
#     # if there is no frame, we have reached the end of the video
#     if not ret:
#         print("End of the video file...")
#         break
#     ############################################################
#     ### Detect the objects in the frame using the YOLO model ###
#     ############################################################
#     # run the YOLO model on the frame
#     results = model(frame)

#     for result in results:
#         # initialize the list of bounding boxes, confidences, and class IDs
#         bboxes = []
#         confidences = []
#         class_ids = []
#         # loop over the detections
#         for data in result.boxes.data.tolist():
#             x1, y1, x2, y2, confidence, class_id = data
#             x = int(x1)
#             y = int(y1)
#             w = int(x2) - int(x1)
#             h = int(y2) - int(y1)
#             class_id = int(class_id)
#             # filter out weak predictions by ensuring the confidence is
#             # greater than the minimum confidence
#             if confidence > conf_threshold:
#                 bboxes.append([x, y, w, h])
#                 confidences.append(confidence)
#                 class_ids.append(class_id)
#                 cv2.circle(frame, (x, y),3, (0, 255, 0), 2)

#     #end = datetime.datetime.now()
#     # calculate the frame per second and draw it on the frame
#     #fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
#     #cv2.putText(frame, fps, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
#     # show the output frame
#     cv2.imshow("Output", frame)
#     # write the frame to disk
    
#     if cv2.waitKey(1) == ord("q"):
#         break

# # release the video capture, video writer, and close all windows
# video_cap.release()

# cv2.destroyAllWindows()