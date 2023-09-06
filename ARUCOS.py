import cv2
import numpy as np
import time 
import tensorflow as tf

def drawsMaker(img, cornerPoint, ids):
    cornerPoint = np.array(cornerPoint, dtype=np.int32)

    for ic, v in enumerate(cornerPoint):
        i = v[0]
        img = cv2.line(img, i[0],i[1], (0,0,255))
        img = cv2.line(img, i[1],i[2], (0,0,255))
        img = cv2.line(img, i[2],i[3], (0,0,255))
        img = cv2.line(img, i[3],i[0], (0,0,255))
        img = cv2.line(img, i[3],i[1], (255,0,55),3)
        img = cv2.putText(img, f"{ids[ic]}", i[2]+[0,20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232,123,0), 2)
    return img


# img = cv2.imread("board.png")
# corners, ids, markerDict = detector.detectMarkers(img)
# # print(corners)
# # img = drawsMaker(img, corners, ids)
# cv2.aruco.drawDetectedMarkers(img, corners, ids)

# print(ids.ravel())
# # print(markerDict[0].distance)
# cv2.imshow("ff", img)
# cv2.waitKey()

def printHello():
    print("hello")
class dataf:
    def __init__(self, fn, rimit) -> None:
        self.fn = fn
        self.distance = rimit
class ArucoMaker:
    def __init__(self, id, newcorners, distance):
        self.id = id
        self.corners = newcorners
        self.distance = distance
        self.centerX = int((self.corners[0][0][0] + self.corners[0][2][0]) * 0.5)
        self.centerY = int((self.corners[0][0][1] + self.corners[0][2][1]) * 0.5)
def arucos(detector, img):
    
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is not None:
        imgs = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    else :
        imgs = img
    # cv2.imshow("ff", imgs)
    # cv2.waitKey(16)

    if ids is None:
        return imgs, None

    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    marker_size = 0.0275
    marker_objpts = np.array([[-marker_size/2,  marker_size/2, 0],
                [ marker_size/2,  marker_size/2, 0],
                [ marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
    camera_matrix = np.eye(3, 3, dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # cv2.solvePnP() 함수를 사용하여 카메라 매개변수를 계산합니다.
    rvec, tvec, _ = cv2.solvePnP(marker_objpts, corners[0], camera_matrix, dist_coeffs)
    # print(tvec)
    makers = {}
    for ic, v in enumerate(ids.ravel()):
        # 계산된 카메라 매개변수를 사용하여 거리를 계산합니다.
        x0 = corners[ic][0][0][1]
        y0 = corners[ic][0][0][0]
        z0 = tvec[0]
        x1 = corners[ic][0][1][1]
        y1 = corners[ic][0][1][0]
        z1 = tvec[1]

        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        makers[v] = ArucoMaker(v, corners[ic], distance)
        # print(v, ":" , distance)

    return imgs, makers

video = cv2.VideoCapture(0)
def command(detector, frame, config:dict):
    """
    최단 거리에 있는 마커의 커맨드를 동작 시킵니다. 
    커맨드는 config dict에 
    { id : dataf(fn, limit) ,  }
    형식으로 입력해주세요.
    """
    global theLook, timeLook
    if not ("tick" in config) :
        raise ValueError
    
    
    maker_img, makers = arucos(detector, frame)
    action = None
    lens = 10000
    if makers is not None:
        for key in config:
            if key in makers:
                if makers[key].distance > config[key].distance:
                    if makers[key].distance < lens:
                        action = key            
                        lens = makers[key].distance
                        
        
    if action is not None:
        if time.time() - timeLook > 3 and theLook:
            config[action].fn()
            timeLook = time.time()
        theLook = False
        return True
    else:
        theLook = True
        return False


config = {
    0 : dataf(printHello, 100),


    "tick" : None
}

model = tf.keras.models.load_model("model.keras")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
theLook = True
timeLook = time.time()

label = {
    0: 0, 1:45, 2:90
}

while True:
    _, f = video.read()
    
    T0F = command(detector, f, config)
    if not T0F:
        pred = model.predict([f])[0]
        deg = label[np.argmax(pred)]
        siri_Ya.write(f"{deg}|{110}".encode())
        

    # ___, asas = arucos(detector, f)
    # if asas is not None:
    #     if 0 in asas:
    #         print(asas[0].distance)

        
    # cv2.imshow("ff", img)
    # cv2.waitKey()
