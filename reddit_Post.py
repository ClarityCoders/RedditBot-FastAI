import praw
import time
import pickle
import requests
import numpy as np
import cv2
from fastai.vision.all import *
import os
from utils.create_token import create_token

# This fixes issue going from Colab to windows!!
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

# Function to Tell FastAI how we want to label data.
# In this case we label it based on parent directories name.
def label_func(x): return x.parent.name

def lets_post(reddit):
    # Load some images to ignore
    img_notfound = cv2.imread(os.path.join(dirname, 'ignore_images/imageNF.png'))
    img_deleted = cv2.imread(os.path.join(dirname, 'ignore_images/DeletedIMG.png'))

    for submission in reddit.subreddit("NSFW_Bot_Playground").new(limit=10000):
        post = True
        print(submission.url)
        try:
            top_level_comments = list(submission.comments)

            for comment in top_level_comments:
                if hasattr(comment, 'body'):
                    if comment.author.name.lower() == "dwigt-snooot":
                        post = False
        except Exception as e:
            print(e)

        if post and ("jpg" in submission.url.lower() or "png" in submission.url.lower()):
            try:
                resp = requests.get(submission.url, stream=True).raw
                
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.resize(image,(224,224))
                # cv2.imshow('Test', image)
                # cv2.waitKey(0)
                
                # Make sure its a valid image
                difference = cv2.subtract(img_notfound, image)
                b, g, r = cv2.split(difference)
                total_difference_NF = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)

                difference = cv2.subtract(img_deleted, image)
                b, g, r = cv2.split(difference)
                total_difference_del = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)

                if total_difference_del > 0 and total_difference_NF > 0:
                    result = learn_inf.predict(image)
                    print("I posted")
                    submission.reply(f"I think this is {result[0].upper()} I'm about {round(result[2][result[1].item()].item() * 100, 2)} sure...\n").mod.distinguish(sticky=True)
            except Exception as e:
                    print(f"Image failed. {submission.url.lower()}")
                    print(e)


# Point to you trained model (pkl file)
# I created a directory called models
dirname = os.path.dirname(__file__)
learn_inf = load_learner(os.path.join(dirname, 'models/export.pkl'))

# Get token file to log into reddit.
# You must enter your....
# client_id - client secret - user_agent - username password
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
else:
    creds = create_token()
    pickle_out = open("token.pickle","wb")
    pickle.dump(creds, pickle_out)

reddit = praw.Reddit(client_id=creds['client_id'],
                    client_secret=creds['client_secret'],
                    user_agent=creds['user_agent'],
                    username=creds['username'],
                    password=creds['password'])

while True:
    print("---Checking for Posts---")
    lets_post(reddit)
    time.sleep(120)
