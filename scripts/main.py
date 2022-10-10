import cv2
import numpy as np

# Cascade Fiter 
fcas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# reading imgages
specs_ori = cv2.imread('imgs/glass.png', -1)
mus_ori = cv2.imread('imgs/mustache.png', -1)

# init and starting camera
campera_init = cv2.VideoCapture(0) 
campera_init.set(cv2.CAP_PROP_FPS, 30)

def Overlay(src, overlay, pos=(0, 0), scale=1):
    """
    Input:
    - src: the camera input (from video feed)
    - overlay: an reziesd image in a cube format
    ----------------
    * The method takes in a few parameters, pos and scale have default values.
    * The overlay gets resized.
    * overlay, and src, gets split up into separate variables with .shape
    * Putting the forground/background inb correct order
    * loop:
        * loop through matrix
        * set alpha 
        * set new src
    * return src
    """
    # resize the overlay 
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)

    # Size of foreground
    h, w, _ = overlay.shape

    # Size- background img
    rows, cols, _ = src.shape

    # pos foreground/overlay img
    y, x = pos[0], pos[1]  

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            # read the alpha channel - controlling the RGB for the img
            alpha = float(overlay[i][j][3] / 255.0)  
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

# Program loop
def Run() -> None:
    while 1:
        ret, img = campera_init.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = fcas.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
    
        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                glass_symin = int(y + 1.5 * h / 5)
                glass_symax = int(y + 2.5 * h / 5)
                sh_glass = glass_symax - glass_symin
 
                li_symin = int(y + 1.5 *h / 6)
                li_symax = int(y + 2.5 * h / 6)
                sh_li = li_symax - li_symin

                mus_symin = int(y + 3.5 * h / 6)
                mus_symax = int(y + 5 * h / 6)
                sh_mus = mus_symax - mus_symin

                face_glass_roi_color = img[glass_symin:glass_symax, x:x + w]
                face_mus_roi_color = img[mus_symin:mus_symax, x:x + w]

                specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
                mustache = cv2.resize(mus_ori, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

                Overlay(face_glass_roi_color, specs)
                Overlay(face_mus_roi_color, mustache)

        cv2.imshow('Test', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): 
            break

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.imwrite('img.jpg', img)
            break

Run()
campera_init.release()
cv2.destroyAllWindows()
