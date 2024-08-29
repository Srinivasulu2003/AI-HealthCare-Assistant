# flask_app.py
import os
from flask import Flask, request
import cv2
import json
import numpy as np
from flask_cors import CORS
import base64
from datetime import datetime
from keras_facenet import FaceNet

app = Flask(__name__)
CORS(app)

# Initialize FaceNet model
embedder = FaceNet()

def img_to_encoding(path, embedder):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (160, 160))  # Resize image to 160x160
    # Get the embedding
    embedding = embedder.embeddings([img])[0]
    return embedding

database = {}

def verify(image_path, identity, database, embedder):
    encoding = img_to_encoding(image_path, embedder)
    dist = np.linalg.norm(encoding - database[identity])
    if dist < 5:
        print(f"It's {identity}, welcome in!")
        match = True
    else:
        print(f"It's not {identity}, please go away.")
        match = False
    return dist, match

@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.get_json()['username']
        img_data = request.get_json()['image64']
        img_path = f'images/{username}.jpg'
        with open(img_path, "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        
        # Register the user by saving their encoding in the database
        database[username] = img_to_encoding(img_path, embedder)
        
        return json.dumps({"status": 200})
    except Exception as e:
        print(f"Error during registration: {e}")
        return json.dumps({"status": 500})

def who_is_it(image_path, database, embedder):
    encoding = img_to_encoding(image_path, embedder)
    min_dist = 1000
    identity = None
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 5:
        print("Not in the database.")
    else:
        print(f"It's {identity}, the distance is {min_dist}.")
    return min_dist, identity

@app.route('/verify', methods=['POST'])
def change():
    img_data = request.get_json()['image64']
    img_name = str(int(datetime.timestamp(datetime.now())))
    img_path = f'images/{img_name}.jpg'
    
    try:
        # Save the image to disk
        with open(img_path, "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))

        # Perform verification
        min_dist, identity = who_is_it(img_path, database, embedder)

        # Return the result
        if min_dist > 5:
            response = json.dumps({"identity": 0})
        else:
            response = json.dumps({"identity": str(identity)})

    except Exception as e:
        print(f"Error during verification: {e}")
        response = json.dumps({"identity": 0})

    finally:
        # Ensure the file is removed
        if os.path.exists(img_path):
            os.remove(img_path)

    return response

if __name__ == "__main__":
    # Create 'images' directory if it does not exist
    if not os.path.exists('images'):
        os.makedirs('images')
        
    app.run(debug=True)
