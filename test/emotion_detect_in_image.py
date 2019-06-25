import cv2
import numpy as np
from keras.models import load_model
import os

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='a utility')

parser.add_argument('-d', '--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')

parser.add_argument('-o', '--out_to_dir',type=str, nargs='?',help='if provided output will be written to csv(semicolon separated) otherwise to stdout. ')
parser.add_argument('-b', '--is_debug', action='store_true', help='A boolean True False')

FLAGS = parser.parse_args()


print(FLAGS)

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/" ]


def to_x_y(strng):
    cords = strng.split('~')

    #return np.array( [ float(cords[0]), float(cords[1]) ] )
    return [ float(cords[0]), float(cords[1]) ]

def is_front_angle(row, allowed_diff):

    try:
        print( "c 31 " + str(row[31]) )
        print(row[31])
        c30 = to_x_y(row[31])
        c2 = to_x_y(row[3])
        c16 = to_x_y(row[17])
        angle_left = abs( c30[0] - c2[0] )      #not angle though ;-)  at this point only distance between points 
        angle_right = abs( c30[0] - c16[0] )    #not angle though ;-)  at this point only distance between points 
        abs_diff = abs( angle_left - angle_right )

        print("the angle diff " + str(abs_diff))
        if abs_diff <= allowed_diff:
            return True 
        else:
            return False 
    except Exception as e:
        print("caught an error in is_front_angle")
        print(e)
        return False
        
def detect_c( path ):

    classifier = load_model('weights/face_model.h5')
    emotion_dict = {'0':'angry','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprise','6':'neutral'}

    cascPath = "haar-classifier/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)


    items = os.listdir( path )
    file_no = 0

    for item in items:

        file_no = file_no + 1
        print(item + " file_no " + str(file_no))

        if item == '.DS_Store':
            continue

        if not '.csv' in item:
            continue

        if os.path.isfile(path+item):
        
            # with open(path+item, newline='') as csvfile:
            #     data = list(csv.reader(csvfile))
            import pandas as pd 
            try:
                data = pd.read_csv(path+item, sep=';')
            except Exception as e:
                print("caught error while reading csv "+item+", skipping.")
                print(e)
                continue

            # print(type(data))
            # print(data)

            if FLAGS.out_to_dir:

                with open( os.path.join( FLAGS.out_to_dir, item ) , 'wb' ) as file:

                    for index, row in data.iterrows():

                        if not is_front_angle(row, 3):
                            print("not a frontal angle face ")
                            continue

                        # filepath 
                        tmpA = row[0].split("~")
                        # im = cv2.imread("__data/headPose.jpg");
                        # im = cv2.imread( "__data/__input_4/__images/" + tmpA[0] )
                        tmpA1 = tmpA[0].split(".avi")
                        filepath = "../opencv_dlib_utils/face-alignment-master/__data/__images/" + tmpA1[0] + ".avi/" + tmpA[0]
                        print( "img path " + filepath )

                        #video_capture = cv2.VideoCapture(0)
                        # video_capture = cv2.VideoCapture("A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2.mp4")     #("A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2.mp4")  #("ronaldo.mp4")
                        disp_emotion = None
                        skip = 0
                        detected_emotions = ""

                        while skip == 0: #video_capture.isOpened():

                            # Capture frame-by-frame
                            # ret, frame = video_capture.read()
                            im_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                            small_frame = cv2.imread(filepath)    #('Pirates_3_smile_h_nm_np1_ri_goo_1.avi15.jpeg') #cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            gray = im_gray #cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                            
                            #
                            # faces = faceCascade.detectMultiScale(
                            #     gray,
                            #     scaleFactor=1.1,
                            #     minNeighbors=5,
                            #     minSize=(30, 30)
                            # )
                            faces = [ [] ]


                            # Draw a rectangle around the faces
                            print( "shpes " )
                            print( small_frame.shape )
                            print( gray.shape )
                            print( faces )
                            for (x, y, w, h) in faces:
                                face = small_frame[y:y+h,x:x+w ,:]
                                face = np.array(face)
                                face = np.resize(face, (48,48,3))
                                face = np.reshape(face ,(1,48,48,3))
                                face = face/255
                                emotion = classifier.predict_classes(face)[0]
                                if(skip%10==0):
                                    skip = 0
                                    disp_emotion = emotion_dict[str(emotion)]

                                if FLAGS.is_debug:                                    
                                    cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(small_frame, disp_emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

                                detected_emotions = disp_emotion + "|"

                            # Display the resulting frame
                            if FLAGS.is_debug:                                    
                                cv2.imshow('image', small_frame)
                                # input("Press Enter to continue...")
                                skip+=1
                                if cv2.waitKey(3000) & 0xFF == ord('q'):
                                    # break
                                    tmp = ''

                            break

                        #write
                        line = "\""+row[0]+"\""
                        for i in range(1, len(row) - 1):
                            line = line + ";\""+ str(row[i]) +"\""

                        line = line + ";\""+detected_emotions+"\"" 

                        file.write(line.encode())
                        file.write('\n'.encode())


for path in paths:
    detect_c( path )
