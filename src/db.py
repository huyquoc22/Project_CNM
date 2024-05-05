from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    check_in_time = Column(DateTime)
    check_out_time = Column(DateTime)
    total_time = Column(String(20))

engine = create_engine('sqlite:///attendance.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def add_user(username):
    session = Session()
    user = User(username=username)
    session.add(user)
    session.commit()
    session.close()

def add_attendance(user_id, check_in_time):
    session = Session()
    attendance = Attendance(user_id=user_id, check_in_time=check_in_time)
    session.add(attendance)
    session.commit()
    session.close()

def update_attendance(user_id, check_out_time, total_time):
    session = Session()
    attendance = session.query(Attendance).filter_by(user_id=user_id, check_out_time=None).first()
    if attendance:
        attendance.check_out_time = check_out_time
        attendance.total_time = total_time
        session.commit()
    session.close()

def get_user_attendance(user_id):
    session = Session()
    attendance = session.query(Attendance).filter_by(user_id=user_id).all()
    session.close()
    return attendance

def add_attendance(name, user_id):
    current_time = datetime.now()

    # Add user to database if not already exists
    add_user(name)

    # Add attendance record
    add_attendance(user_id, current_time)

def update_attendance(user_id):
    current_time = datetime.now()
    user_attendance = get_user_attendance(user_id)

    if user_attendance:
        last_attendance = user_attendance[-1]
        check_in_time = last_attendance.check_in_time
        total_time = current_time - check_in_time
        update_attendance(user_id, current_time, total_time)
    else:
        print("User not found in attendance records.")
