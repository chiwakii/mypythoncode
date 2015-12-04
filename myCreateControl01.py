from create2.create2 import Create2
import time
import cv2 as cv

def rotate(angle,speed):
        correction = 0.9917
        total_angle = create.get_angle()
        if angle => 0:
                create.drive(speed,1)
                time.sleep(0.1)
                while create.get_angle() < (total_angle + angle) * correction:
                        print(create.get_angle())
                        time.sleep(0.1)
        elif angle < 0:
                create.drive(speed,-1)
                time.sleep(0.1)
                while -1 * create.get_angle() < -(total_angle + angle) * correction:
                        print(create.get_angle())
                        time.sleep(0.1)
        create.drive(0,0)
        time.sleep(0.1)
        return create.get_angle()

def stright(speed,distance):
        correction = 0.926
        create.drive(speed,0)
        while create.get_distance() < distance*correction:
                time.sleep(0.1)
        score = create.get_distance()
        create.drive(0,0)
        print("clear")
        return score

def curve(right_s,left_s,distance):
        correction = 0.926
        create.drive_wheels(right_s,left_s)
        while create.get_distance() < distance*correction:
                time.sleep(0.1)
        score = create.get_distance()
        create.drive(0,0)
        print("clear")
        return score

##main##
create = Create2(threading=True)
create.start()
time.sleep(0.1)

t_angle = rotate(20,30)
t_distance = stright(100,1000)
t_distance = curve(100,200,t_distance+500)
stright(100,t_distance+500)
cap = cv.VideoCapture(0)

while 1:
    ret,frame = cap.read()
     
    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('a'):
        rotate(10,30)
    elif k == ord('s'):
        stright(100,100)
cv.destroyAllWindows()
cap.release()
