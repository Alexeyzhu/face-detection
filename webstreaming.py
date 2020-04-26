# USAGE
# python webstreaming.py (optional) --ip 0.0.0.0 --port 8000

import argparse
import datetime
import threading
import time

import cv2
import imutils
from flask import Flask
from flask import Response
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from flask_table import Table, Col

# import the necessary packages
from facerecognizer.face_recognition import *

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'True'
db = SQLAlchemy(app, )


class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    embedding = db.Column(db.String, unique=True, nullable=False)
    person_id = db.Column(db.Integer,
                          db.ForeignKey('person.id'), nullable=False)


class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String, unique=True, nullable=False)

    def __repr__(self):
        return f'ID:{self.id} Name:{self.name}'


class Presence(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp_first = db.Column(db.DateTime)
    timestamp_last = db.Column(db.DateTime)
    person_id = db.Column(db.Integer,
                          db.ForeignKey('person.id'), nullable=False)

    def __repr__(self):
        p = Person.query.filter_by(id=self.person_id).first()
        return {"name": p.name, "timestamp_first": self.timestamp_first, "timestamp_last": self.timestamp_last}


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting web camera stream...")
vs = WebcamStream().start()
fps = vs.get(cv2.CAP_PROP_FPS)
print(f'[INFO] camera FPS: {fps}')
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


class ResultsTable(Table):
    classes = ['table', 'table-bordered', 'table-striped']
    name = Col('Name')
    timestamp_first = Col('Timestamp first')
    timestamp_last = Col('Timestamp last')

@app.route('/table')
def show_table():
    prs = Presence.query.all()
    items = []
    for presence in prs:
        items.append(presence.__repr__())
    table = ResultsTable(items)
    return render_template('table.html', table=table)


@app.route("/retrain")
def retrain():
    ee = ExtractEmbeddings()
    try:
        ee.process(db)
        return render_template("succ.html")
    except Exception:
        return render_template("fail.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def face_recognition():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the face recognizer and the total number of frames
    # read thus far
    fr = FaceRecognizer()
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        grab, frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio)
        frame = imutils.resize(frame, width=600)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        frame, names = fr.recognize(frame=frame)

        for name in names:
            person_id = Person.query.filter_by(name=name).first().id
            presence = Presence.query.filter_by(person_id=person_id).first()

            if presence is not None:
                presence.timestamp_last = timestamp
            else:
                new_presence = Presence(person_id=person_id,
                                        timestamp_first=timestamp, timestamp_last=timestamp)

                db.session.add(new_presence)
            db.session.commit()

        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", default="127.0.0.1",
                    type=str, help="ip address of the device")
    ap.add_argument("-o", "--port", default=5000,
                    type=int, help="port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    db.drop_all()
    db.create_all()

    ee = ExtractEmbeddings()
    ee.process(db)

    # start a thread that will perform motion detection
    t = threading.Thread(target=face_recognition)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.release()
