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

def straight(speed,distance):
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


def Drive(x,y,speed):
        self.angle = create.get_angle()
        self.encl = create2.get_left_encoder()
        self.encr = create2.get_right_encoder()
        #self.angle = 0.036*0.5*(1./0.1175)*(self.encl-self.encr)
        print("1st:Rotate Start" + str(self.angle) )
        rotate(pi/2 - atan2(y,x) - self.angle, speed)
        print("2st:Straight Start" + str(self.encl) + str(self.encr) )
        straight(speed,sqrt(x**2 + y**2))
        print("Fin:Drive end")

##main##
create = Create2(threading=True)
create.start()
time.sleep(0.1)

while True:
        k = cv.waitKey(30) & 0xff
        if k == 27 or k == ord('q'):
                break
        elif k == ord('a'):
                rotate(10,30)
        elif k == ord('s'):
                stright(100,100)
        elif k == ord('f'):
                Drive(100,100,100)
