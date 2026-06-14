import streamlit as st
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pandas as pd

# Constants
FACULTY_DIR = "faculty_images"
LOG_DIR = "attendance_logs"

def get_today_log_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"Attendance_{today}.csv")

@st.cache_data
def load_known_faces():
    """Load known faculty images and encode them."""
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(FACULTY_DIR):
        os.makedirs(FACULTY_DIR)

    for filename in os.listdir(FACULTY_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            path = os.path.join(FACULTY_DIR, filename)
            # Load the image
            image = face_recognition.load_image_file(path)
            # Encode the face (assuming 1 face per image)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                # Remove extension to get the name
                name = os.path.splitext(filename)[0]
                # Replace underscores with spaces for better display
                name = name.replace("_", " ")
                known_face_names.append(name)
            else:
                st.warning(f"No face found in {filename}. Skipping.")
    return known_face_encodings, known_face_names

def mark_attendance(name):
    """Mark attendance in today's CSV file."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    log_file = get_today_log_file()
    
    # Check if file exists to write headers
    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=["Name", "Status", "Time", "Date"])
        df.to_csv(log_file, index=False)
    
    # Read current data
    df = pd.read_csv(log_file)
    
    # If the person is already marked present today, do nothing
    if name not in df["Name"].values:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        # Append new record
        new_record = pd.DataFrame([{"Name": name, "Status": "Present", "Time": time_str, "Date": date_str}])
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(log_file, index=False)
        return True
    return False

def main():
    st.set_page_config(page_title="Faculty Attendance System", page_icon="🏫", layout="wide")
    
    st.title("🏫 Facial Recognition Attendance System")
    st.markdown("Welcome! Please stand in front of the camera to mark your attendance.")

    # Sidebar for controls and logs
    st.sidebar.title("Controls")
    run_camera = st.sidebar.checkbox("Start Camera")

    st.sidebar.title("Today's Attendance")
    log_file = get_today_log_file()
    
    attendance_placeholder = st.sidebar.empty()
    
    def update_attendance_table():
        if os.path.isfile(log_file):
            df = pd.read_csv(log_file)
            attendance_placeholder.dataframe(df, width=True)
        else:
            attendance_placeholder.info("No attendance recorded yet today.")

    update_attendance_table()

    # Load faces
    with st.spinner("Loading faculty profiles..."):
        known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_names:
        st.warning(f"No faculty images found. Please add `.jpg` or `.png` images to the `{FACULTY_DIR}` directory.")
        return

    st.success(f"Loaded {len(known_face_names)} faculty profiles.")

    # Main area for video feed
    FRAME_WINDOW = st.image([])

    if run_camera:
        # Open webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            st.error("Error: Could not access the camera.")
            return

        while run_camera:
            # Read a single frame of video
            ret, frame = video_capture.read()
            
            if not ret:
                st.error("Failed to grab frame.")
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        # Mark attendance
                        if mark_attendance(name):
                            st.toast(f"✅ Attendance marked for {name}!")
                            update_attendance_table()

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1)

            # Convert frame back to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame)

        # Release handle to the webcam
        video_capture.release()
    else:
        st.info("Check 'Start Camera' in the sidebar to begin attendance.")

if __name__ == "__main__":
    main()
