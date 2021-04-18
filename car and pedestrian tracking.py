import cv2
from random import randrange

car_image = cv2.imread("car image.jpg")
vid = cv2.VideoCapture("y2mate.com - Pedestrians Compilation_480p.mp4")

#our pre_trained car classifier
car_tracker_file = "car_detector.xml"
pedestrian_tracker_file = "haarcascade_fullbody.xml"

#creating car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    success,frame = vid.read()

    #creating car classifier
   # car_tracker = cv2.CascadeClassifier(car_tracker_file)
    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)
    #convert image to grayscale
    black_white = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #detecting car and pedestrian
    car = car_tracker.detectMultiScale(black_white)
    #detect pedestraian
    pedestrian = pedestrian_tracker.detectMultiScale(black_white)
    
    print(pedestrian)
    #drawing rectangle for car
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x + w, x + h), (255, 0, 0), 2)
        
    #drawing rectangle around pedestrian
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x + w, x + h), (0, 255, 0), 2)

    #displaying image
    cv2.imshow("image of car",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)

#convert image to grayscale
#black_white = cv2.cvtColor(car_image,cv2.COLOR_BGR2GRAY)
#black_white = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)

#detecting car
#car = car_tracker.detectMultiScale(black_white)
#pedestrian = car_tracker.detectMultiScale(black_white)

#print(car)
#drawing rectangle for car
#for (x, y, w, h) in car:
    cv2.rectangle(black_white, (x, y), (x + w, x + h), (randrange(256), randrange(256), randrange(256)), 2)

#displaying image
#cv2.imshow("image of car",black_white)
#cv2.waitKey(0)



""""


#cars = cv2.VideoCapture("y2mate.com - Best Opening Races From Pixars Cars  Pixar Cars_480p.mp4")
cars = cv2.VideoCapture("y2mate.com - Tesla Dash Cam moments from Sweden  TESLACAM STORIES 10_480p (1).mp4")
#our pre_trained car classifier
clasifier_file = "car_detector.xml"
#run for ever until car stops
#while True:
    success,frame = cars.read()


    #creating car classifier
    car_tracker = cv2.CascadeClassifier(clasifier_file)
    #convert image to grayscale
    black_white = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #detecting car
    car = car_tracker.detectMultiScale(black_white)
    print(car)
    #drawing rectangle for car
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x + w, x + h), (randrange(256), randrange(256), randrange(256)), 2)

    #displaying image
    cv2.imshow("image of car",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
"""





