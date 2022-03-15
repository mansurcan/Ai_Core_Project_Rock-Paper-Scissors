from importlib.resources import path
import platform
import random
from unittest import result
import cv2
from shutil import move
from keras.models import load_model
import numpy as np
import tensorflow as tf




def calculate_winner():
    # Load the model, open the cam, resize the image
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    cap = cv2.VideoCapture(0)
    move2 = random.choice(['rock','paper','scissors'])
    
    while True:
        
        ret, frame = cap.read()
        resized_frame = cv2.resize(
            frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(
            np.float32) / 127.0) - 1  # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)
        
        if prediction[0][0] > 0.5:
            print('nothing')
            move1 = 'nothing'
            
        elif prediction[0][1] > 0.5:
            print('rock')
            move1 = 'rock'
           
            if move1 == "rock" and move2 == "scissors":
                # if move2 == "scissors":
                    result = "User"
                    print("The winner is " + result)

            elif move2 == "paper":
                result =  "Computer"
                print("The winner is " + result)
                
            else: 
                move2 == "rock"
                result = "Tie"
                print("The winner is " + result)

            
        elif prediction[0][2] > 0.5:
            print('paper')
            move1 = 'paper'
            if move1 == "paper":
                if move2 == "rock":
                    result =  "User"
                    print("The winner is " + result)

                elif move2 == "scissors":
                     result =  "Computer"
                     print("The winner is " + result)
                     
                else: 
                    move2 == "paper"
                    result = "Tie"
                    print("The winner is " + result)
            
        elif prediction[0][3] > 0.5:
            print('scissors')
            move1 = 'scissors'
            if move1 == "scissors":
                if move2 == "paper":
                    result =  "User"
                    print("The winner is " + result)

                elif move2 == "rock":
                    result =  "Computer"
                    print("The winner is " + result)
                    
                else: 
                    move2 == "scissors"
                    result = "Tie"
                    print("The winner is " + result)

            

        # Press q to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    print('Python version:', platform.python_version())
    print('Tensorflow version:', tf.__version__)
    print('Keras version:', tf.keras.__version__)
    calculate_winner()
pass
