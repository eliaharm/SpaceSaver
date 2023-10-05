'''
Recognize and blur all faces in photos.
'''
import os
import sys
import cv2
# import face_recognition
# importing numpy as geek  
import numpy as np 

# to detect the face of the human 
# cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
cascadeFront = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml") 
cascadeProfile = cv2.CascadeClassifier("lbpcascade_profileface.xml") 

def face_blur(src_img, dest_img, zoom_in=1):
    '''
    Recognize and blur all faces in the source image file, then save as destination image file.
    '''
    sys.stdout.write("%s:processing... \r" % (src_img))
    sys.stdout.flush()

    # Initialize some variables
    face_locations = []
    
    photo = cv2.imread(src_img)
    # convert the frame into grayscale(shades of black & white) 
    gray_image = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 

    # detect multiple faces in a captured frame 
    # scaleFactor: Parameter specify how much the 
    # image sizeis reduced at each image scale. 
    # minNeighbors: Parameter specify how many 
    # neighbours each rectangle should have to retain it. 
    # rectangle consists the detect object. 
    # Here the object is the face. 
    face_locationsFront = cascadeFront.detectMultiScale( 
		  gray_image, scaleFactor=1.05,
      minNeighbors=20, minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE)
    face_locationsProfile = cascadeProfile.detectMultiScale( 
		  gray_image, scaleFactor=1.05,
      minNeighbors=20, minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE) 
    face_locations = np.concatenate((face_locationsFront , face_locationsProfile), axis = 0)
    
    if hasattr(face_locationsFront,'size'):
        print("%s:There are %s faces at " % (src_img, len(face_locationsFront)), face_locationsFront)
    else:
        print('%s:There are no faces in the photo.' % (src_img))
        return False
    if hasattr(face_locationsProfile,'size'):
        print("%s:There are %s faces at " % (src_img, len(face_locationsProfile)), face_locationsProfile)
    else:
        print('%s:There are no faces in the photo.' % (src_img))
        return False
    if hasattr(face_locations,'size'):
        print("%s:There are %s faces at " % (src_img, len(face_locations)), face_locations)
    else:
        print('%s:There are no faces in the photo.' % (src_img))
        return False

    #Blur all face
    photo = cv2.imread(src_img)
    # for top, right, bottom, left in face_locations:
    for x, y, w, h in face_locations: 
        # Scale back up face locations since the frame we detected in was scaled to 1/zoom_in size
        # top *= zoom_in
        # right *= zoom_in
        # bottom *= zoom_in
        # left *= zoom_in
        print("processing... x:%s\r" % (x))

        # Extract the region of the image that contains the face
        # face_image = photo[top:bottom, left:right]
        face_image = photo[y:(y+h), x:(x+w)]

        cv2.imwrite(dest_img+str(x)+".jpg", face_image)
        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (21, 21), 0)

        # Put the blurred face region back into the frame image
        # photo[top:bottom, left:right] = face_image
        photo[y:(y+h), x:(x+w)] = face_image

    #Save image to file
    cv2.imwrite(dest_img, photo)

    print('Face blurred photo has been save in %s' % dest_img)

    return True

def blur_all_photo(src_dir, dest_dir):
    '''
    Blur all faces in the source directory photos and copy them to destination directory
    '''
    src_dir = os.path.abspath(src_dir)
    dest_dir = os.path.abspath(dest_dir)
    print('Search and blur human faces in %s''s photo.' % src_dir)
    for root, subdirs, files in os.walk(src_dir):
        root_relpath = os.path.relpath(root, src_dir)
        new_root_path = os.path.realpath(os.path.join(dest_dir, root_relpath))
        os.makedirs(new_root_path, exist_ok=True)

        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.jpg':
                srcfile_path = os.path.join(root, filename)
                destfile_path = os.path.join(new_root_path, os.path.basename(filename))
                face_blur(srcfile_path, destfile_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('faceblur v1.0.0 (c) telesoho.com')
        print('Usage:python faceblur <src-image/src-directory> <dest-image/dest-directory>')
    else:
        if os.path.isfile(sys.argv[1]):
            face_blur(sys.argv[1], sys.argv[2])
        else:
            blur_all_photo(sys.argv[1], sys.argv[2])
