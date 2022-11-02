import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
X_train,y_train,X_test,y_test = train_test_split(X,y,test size  = 2500,random_state = 9,train_size = 7500)
X_train_scale = X_test/255
X_test_scale = X_test/255
lr = LogisticRegression(solver = 'saga',multi_class = 'multinomial')
lr.fit(X_train_scale,y_train)

def get_pred(image):
    im_pil = Image.open(image)
    img_bw = im_pil.convert('L')
    img_bw_resized = img_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized,pixel_filter)
    img_bw_resized_inverted_scale = np.clip(img_bw_resized-min_pixel,0,255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scale = np.asarray(img_bw_resized_inverted)/max_pixel
    test_sample = np.array(img_bw_resized_inverted).reshape(1,784)
    test_pred = lr.predict(test_sample)
    return test_pred[0]

