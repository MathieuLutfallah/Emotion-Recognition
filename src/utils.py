def apply_cnn_detection(img,cnn_face_detector):

    cnn_dets = cnn_face_detector(img, 1)
    dets = dlib.rectangles()
    dets.extend([d.rect for d in cnn_dets])
    return dets

    

def face_alignment(imageTaken,detector,cnn_face_detector,sp):
    # Load the image using OpenCV
    bgr_img = imageTaken
    if bgr_img is None:
        print("Sorry, we could not load '{}' as an image".format(face_file_path))
        exit()

    # Convert to RGB since dlib uses RGB images
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    save_img = img.copy()
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    ''' traditional method '''
    dets = detector(img, 1)
    if len(dets) == 0:
        # first use cnn detector
        dets = apply_cnn_detection(img,cnn_face_detector)
        if len(dets) == 0:
            ''' Linear '''
            img = LinearEqual(img)
            dets = apply_cnn_detection(img,cnn_face_detector)
            if len(dets) == 0:
                ''' clahe '''
                img = claheColor(img)
                dets = apply_cnn_detection(img,cnn_face_detector)
                if len(dets) == 0:
                    #''' Histogram_equalization '''
                    img = hisEqulColor(img)
                    dets = apply_cnn_detection(img,cnn_face_detector)
                    if len(dets) == 0:
                        return None

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()

    for detection in dets:
        faces.append(sp(img, detection))
    image = dlib.get_face_chip(save_img, faces[0], size=224, padding=0.25)
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return cv_bgr_img


