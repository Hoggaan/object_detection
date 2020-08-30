
#### Pet detection function ####

# This function contains the code to detect a pet, determine if it's
# inside or outside, and send a text to the user's phone.
def pet_detector(frame):

    # Use globals for the control variables so they retain their value after function exits
    global detected_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    
    # Check the class of the top detected object by looking at classes[0][0].
    # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
    # find its center coordinates by looking at the boxes[0][0] variable.
    # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
    if (((int(classes[0][0]) == 17) or (int(classes[0][0] == 18) or (int(classes[0][0]) == 88))) and (pause == 0)):
                
    return frame
