import cv2


def measurement(frame, e1):
    time = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    fps = "FPS:{0:0.2f}".format(1/time)
    frame = cv2.putText(frame, fps, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("frame", frame)
    #print(fps)
    return 0


def putMsg(frame, armor, ns):
    ns[2] += 1
    ns[4] += 1
    ns[1] = ns[3] / ns[2]
    if len(armor) > 0:
        for x, y, w, h in armor:
            cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 0, 255), 2)
            image = frame[y:y+h, x:x+w]
            image = cv2.resize(image, (150, 50))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("image", image)
            #print(x,y,w,h)
        ns[3] += 1
        ns[5] += 1
    if ns[2]%ns[6] is 0:
        ns[0]  = ns[5] / ns[4]
        ns[4] = ns[5] = 0
    cv2.putText(frame, "intime:{0:0.2f}  alltime:{1:0.2f}".format(
        ns[0], ns[1]), (250, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    return ns[2]
