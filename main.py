import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask,request,render_template
import cv2
import os
import pathlib
import h5py

app=Flask(__name__)

@app.route("/")
def home():
   return render_template("home.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        f=request.files["myfile"]
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            
           basepath,"uploads")
            
        f.save(file_path)
        
        file=open("model/my_model.json")
        load_model_json=file.read()
        file.close()
        
        model=keras.models.model_from_json(load_model_json)
        model.load_weights("model/my_model.h5")
        
        
        img_arr=cv2.imread(str(file_path))
        resized_img=cv2.resize(img_arr,(224,224))
        x=np.array(resized_img)
        x=x/255
        x=x.reshape(1,224,224,3)
        
        model.compile(optimizer="SGD",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        
        data=np.argmax(model.predict(x))
        
        if data==1:
            d="you're suffering from Viral Pneumonia"
        elif data==2:
            d="you're suffering from covid-19"
        else:
            d="you,re normal"
        
        return render_template("home.html",data=d)
        
    else:
        return("something went wrong")
        
        
        
        
        
        
    






if __name__=="__main__":   
    app.run("localhost","8080",use_reloader=False,debug=True)