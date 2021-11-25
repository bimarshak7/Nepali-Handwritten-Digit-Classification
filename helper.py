import numpy as np
import base64
#from io import BytesIO
#from PIL import Image, ImageOps
import cv2

Theta1 = np.load('Theta1.npy')
Theta2 = np.load('Theta2.npy')
digits = {0:'0',1:'१',2:'२',3:'३',4:'४',5:'५',6:'६',7:'७',8:'८',9:'९'}

def sigmoid(z):
        return 1/(1 + np.exp(-z))
def forward_pass(x_train):
        '''Calculate op using forward propagation'''
        m = x_train.shape[0]
        # input layer activations becomes sample
        A1 = np.concatenate([np.ones((1)), x_train])

        # input layer to hidden layer 1
        Z1 = np.dot(Theta1, A1)
        A2 = np.concatenate([np.ones(1), sigmoid(Z1)])

        # hidden layer to op layer
        Z2 = np.dot(Theta2, A2)
        A3 = sigmoid(Z2)
        return A3.argmax(),A3
def processImage(img):
        _, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        [xl,yl,wl,hl] = cv2.boundingRect(contours[0])
        maxar=hl*wl
        for contour in contours:
            # get rectangle bounding contour
                [x,y,w,h] = cv2.boundingRect(contour)
                if (w*h)>maxar:[xl,yl,wl,hl]=[x,y,w,h]
            # draw rectangle around contour on original image
        #cv2.rectangle(image,(xl,yl),(xl+wl,yl+hl),(255,0,255),2)
        cropped_image = binary[yl:yl+hl,xl:xl+wl]
        final=cv2.resize(cropped_image, (32,32), interpolation= cv2.INTER_LINEAR)
        return final

def predict(im_data):
        im_bytes = base64.b64decode(im_data.split(',')[1])
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        cv2.imwrite('image.png',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        op = processImage(gray)
        cv2.imwrite('op.png',op)
        #resized = cv2.resize(gray, (32,32), interpolation= cv2.INTER_LINEAR)
        #cv2.imshow("IMG",resized)
        #resized=~resized
        #resized=resized/255
        #resized = cv2.convertTo(resized, CV_32FC3, 1/255);
        #print("Max:",resized)
        res,prob = forward_pass(op.flatten()/255)
        #print("Result:",res)
        #np.save('samp2.npy',array)
        return str(digits[res])