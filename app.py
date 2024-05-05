import cv2
import os
from flask import Flask,request,render_template,redirect,url_for,session
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil

#### Defining Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,ID,Check_in_time,Check_out_time,Total_time')
if not os.path.isdir(f'Attendance/Attendance_faces-{datetoday}'):
    os.makedirs(f'Attendance/Attendance_faces-{datetoday}')




#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### get name and id of all users
def getusers():
    nameUsers = []
    idUsers = []
    l = len(os.listdir('static/faces'))
    for user in os.listdir('static/faces'):
        nameUsers.append(user.split('_')[0])
        idUsers.append(user.split('_')[1])
    return nameUsers, idUsers, l

#### delete user
def delUser(userid, username):
    for user in os.listdir('static/faces'):
        if user.split('_')[1] == userid:
            shutil.rmtree(f'static/faces/{username}_{userid}', ignore_errors=True)

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['ID']
    inTimes = df['Check_in_time']
    outTimes = df['Check_out_time']
    totalTimes = df['Total_time']
    l = len(df)
    return names,rolls,inTimes,outTimes,totalTimes,l

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time},'',''')
    else:
        row_index = 0

        for i in range(0, df['ID'].count()):
            if str(df['ID'][i]) == userid:
                row_index = i
                break

        if str(df['Check_out_time'][row_index]) == 'nan':
            df.loc[row_index, 'Check_out_time'] = current_time

            inTime = (datetime.strptime(df['Check_in_time'][row_index], '%H:%M:%S'))
            outTime = (datetime.strptime(df['Check_out_time'][row_index], '%H:%M:%S'))

            totalTime = outTime - inTime

            df.loc[row_index, 'Total_time'] = totalTime

            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

#Get check in and out time of user    
def getUserTime(userid):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    row_index = 0

    for i in range(0, df['ID'].count()):
        if str(df['ID'][i]) == userid:
            row_index = i
            break
            
    return str(df['Check_in_time'][row_index]), str(df['Check_out_time'][row_index])

#Check existed userID
def checkUserID(newuserid):
    listID = os.listdir('static/faces')
    for i in range(0, len(listID)):
        if listID[i].split('_')[1] == newuserid:
            return True
    return False

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def base():
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
    return render_template('base.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

#### Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True  # Set the session variable to indicate login
            return redirect(url_for('home'))
        else:
            return render_template('login.html', mess='Invalid credentials')

    return render_template('login.html', error=None)

#### Home Admin Page
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes,
                           totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Logout function
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove the 'logged_in' session variable
    return redirect(url_for('base'))

#### List all users
@app.route('/listUsers')
def users():
    names, rolls, l = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l)

#### Delete user
@app.route('/deletetUser', methods=['POST'])
def deleteUser():
    userid = request.form['userid']
    username = request.form['username']
    delUser(userid, username)
    names, rolls, l = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l, mess='Delete successfully.')

#### This function will run when we click on Check in / Check out Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('base.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    face_detected = False
    identified_person = None

    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            face_detected = True

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not face_detected:
        print("No face detected.")

    if identified_person is not None:
        add_attendance(identified_person)

        # Assuming extract_attendance() retrieves these values
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()

        # Check if all variables have values before rendering template
        if names is not None and rolls is not None and inTimes is not None and outTimes is not None and totalTimes is not None and l is not None:
            return render_template('base.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2)
        else:
            # Handle case where data is not available
            return render_template('base.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Attendance data not available.')
    else:
        # Handle case where no person was identified
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('base.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No person identified.')

    # Handle default case
    return render_template('base.html', totalreg=totalreg(), datetoday2=datetoday2, mess='Unknown error.')

@app.route('/admin',methods=['GET'])
def admin():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    face_detected = False
    identified_person = None

    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            face_detected = True

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not face_detected:
        print("No face detected.")

    if identified_person is not None:
        add_attendance(identified_person)

        # Assuming extract_attendance() retrieves these values
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()

        # Check if all variables have values before rendering template
        if names is not None and rolls is not None and inTimes is not None and outTimes is not None and totalTimes is not None and l is not None:
            return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2)
        else:
            # Handle case where data is not available
            return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Attendance data not available.')
    else:
        # Handle case where no person was identified
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No person identified.')

    # Handle default case
    return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='Unknown error.')

#### This function will run when we add a new user
@app.route('/addUsers', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        if 'newusername' in request.form and 'newuserid' in request.form:
            newusername = request.form['newusername']
            newuserid = request.form['newuserid']

            if checkUserID(newuserid) == False:
                userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
                if not os.path.isdir(userimagefolder):
                    os.makedirs(userimagefolder)

                cap = cv2.VideoCapture(0)
                i, j = 0, 0
                face_detected = False

                while 1:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    faces = extract_faces(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        cv2.putText(frame, f'Images Captured: {i}/100', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 20), 2, cv2.LINE_AA)
                        if j % 10 == 0:
                            name = newusername + '_' + str(i) + '.jpg'
                            cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                            i += 1
                            face_detected = True

                        j += 1
                    if j == 1000:
                        break

                    cv2.imshow('Adding new User', frame)
                    if cv2.waitKey(1) == 27:
                        break

                cap.release()
                cv2.destroyAllWindows()

                if not face_detected:
                    delUser(newuserid, newusername)
                    return render_template('addUser.html', totalreg=totalreg(), datetoday2=datetoday2, mess='No person identified.')
                print('Training Model')
                train_model()

                names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
                return render_template('addUser.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes,
                                       totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2,mess1='Add successfully.')

            else:
                names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
                return render_template('addUser.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes,
                                       totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2,
                                       mess ='User ID already exists. Please choose another ID.')
        else:
            return "Invalid form submission. Please provide 'newusername' and 'newuserid'."

    else:
        return render_template('addUser.html', totalreg=totalreg(), datetoday2=datetoday2)

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)