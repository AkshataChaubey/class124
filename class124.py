import cv2
import numpy as np

import tensorflow as tf

mymodel=tf.keras.models.load_model("keras_model.h5")
print(mymodel)
video=cv2.VideoCapture(0)
while True:
    dummy,frame=video.read()

    myimage=cv2.resize(frame(224,224))


    test_image=np.array(myimage,dtype=np.float32)
   
    print(test_image)
    expand_image=np.expand_dims(test_image,axis=0)
    # print(expand_image)

    normalised_image=expand_image/255.0
    # print(normalised_image)

    predication=mymodel.predict(normalised_image)
    print("predication:  ",predication)

    
    cv2.imshow("window",frame)
    key=cv2.waitKey(1)
    if key==32:
        print("closing")
        break
video.release()
cv2.destroyAllWindows()    









