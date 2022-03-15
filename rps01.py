import platform
import random
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf


def calculate_winner():
    # Load the model, open the cam, resize the image
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    cap = cv2.VideoCapture(0)
    computer = random.choice(['rock','paper','scissors'])
    
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
            print('User: nothing')
            user = 'nothing'
            computer = 'nothing'
            
        elif prediction[0][1] > 0.5:
            print('User: rock')
            user = 'rock'           
            if user == "rock" and computer == "scissors":
                result = "User"
                print("The winner is " + result)

            elif user == "rock" and computer == "paper":
                result =  "Computer"
                print("The winner is " + result)
                
            else: 
                user == "rock" and computer == "rock"
                result = "Tie"
                print("The winner is " + result)

            
        elif prediction[0][2] > 0.5:
            print('User: paper')
            user = 'paper'
            if user == "paper" and computer == "rock":
                    result =  "User"
                    print("The winner is " + result)

            elif user == "paper" and computer == "scissors":
                    result =  "Computer"
                    print("The winner is " + result)
                     
            else: 
                user == "paper" and computer == "paper"
                result = "Tie"
                print("The winner is " + result)
            
        elif prediction[0][3] > 0.5:
            print('User: scissors')
            user = 'scissors'
            if user == "scissors" and computer == "paper":
                result =  "User"
                print("The winner is " + result)

            elif user == "scissors" and computer == "rock":
                result =  "Computer"
                print("The winner is " + result)
                    
            else: 
                user == "scissors" and computer == "scissors"
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
