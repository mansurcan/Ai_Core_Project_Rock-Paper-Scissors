import random
import time
import cv2
from keras.models import load_model
import numpy as np

# Countdown
countdown = False
current_time = 0
time_calculation = 0

begin = False
user_first = None
total_attempts = 3
user_points = 0
computer_points = 0

# User teachable machine model prediction results 
prediction_results = {0: "nothing", 1: "rock", 2: "paper", 3: "scissors"}

def map(val):
    return prediction_results[val]

def calculate_winner(player1, player2):
    if player1 == player2:
        return "Tie"
    
    if player1 == "rock":
        if player2 == "scissors":
            return "user"
        if player2 == "paper":
            return "computer"

    if player1 == "paper":
        if player2 == "rock":
            return "user"
        if player2 == "scissors":
            return "computer"

    if player1 == "scissors":
        if player2 == "paper":
            return "user"
        if player2 == "rock":
            return "computer"

# Load the model, open the cam, resize the image
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)  

# Continuosly go through the predition until user or computer gets 3 points
while True:
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1  # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data) 
    
    if not begin:
        cv2.putText(frame, f"Press s to start!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

    if cv2.waitKey(20) == ord('s'):
        if not begin:
            current_time = time.time()
            begin = True
            countdown = True 
    if begin:
        time_calculation = 4 - (time.time() - current_time)
        if time_calculation <= -3:
            
            cv2.putText(frame, f"Press p to play the next round!", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if cv2.waitKey(20) == ord('p'):
                begin = False
                countdown = False
                time_calculation = 0
                
        elif time_calculation <= 0:
            countdown = False

            # User's choice
            user_value = np.argmax(prediction[0])
            user = map(user_value)
            
            # Region on the screen for computer to show its hand
            cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
            
            # Calculate the winner 
            if user_first != user:
                if user != "nothing":
                    computer = random.choice(['rock', 'paper', 'scissors'])
                    winner = calculate_winner(user, computer)
                    total_attempts -= 1
                    if winner == "user":
                        user_points += 1
                    elif winner == "computer":
                        computer_points += 1
                    elif winner == "tie":
                        pass
                else:
                    computer = "nothing"
                    winner = "Waiting..."
            user_first = user
            
            # Print the results
            print(f'user: {user} vs. computer: {computer} winner: {winner}')         
            print(f'user points: {user_points} vs. computer points: {computer_points}')
            
            # The region of image to be detected
            region = frame[100:500, 100:500]
            image = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Display the choice of computer
            icon = cv2.imread("images/{}.png".format(computer))
            icon = cv2.resize(icon, (400, 400))
            frame[100:500, 800:1200] = icon
            
            # Display the information
            cv2.putText(frame, f"Winner: {winner}", (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f"User Move: {user} User point: {user_points}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Computer Move: {computer} Computer point: {computer_points}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if total_attempts == 0:
                begin = True
            
    # Display the countdown
    if countdown:
        cv2.putText(frame, f'Show your hand in:', (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f'{int(time_calculation)}', (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 4, cv2.LINE_AA)

    # Press q to close the window
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
            
    # Show the frame
    cv2.imshow(' ----------- Rock Paper Scissors Game by Mansur Can ----------- ', frame)

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


    

