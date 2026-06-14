from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import cv2
import face_recognition
from typing import Optional
import os
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Face Verification API", description="API to verify face images against Firebase Firestore embeddings")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize Firebase
try:
    # Support for Hugging Face Secrets via environment variable
    firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds:
        cred = credentials.Certificate(json.loads(firebase_creds))
        firebase_admin.initialize_app(cred)
    else:
        # Fallback to local file for development
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Warning: Firebase initialization failed. Ensure serviceAccountKey.json exists or FIREBASE_CREDENTIALS secret is set. Error: {e}")
    db = None

class VerificationResponse(BaseModel):
    verified: bool
    name: Optional[str] = None
    club: Optional[str] = None
    message: str

def calculate_distance(embedding1, embedding2):
    """Calculate Euclidean distance between two face embeddings."""
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

import asyncio

def process_image_and_match(contents: bytes) -> VerificationResponse:
    # 2. Convert raw bytes into an image that face_recognition can read
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return VerificationResponse(verified=False, message="Error: Invalid image file.")
        
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Extract the embedding on the server
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) != 1:
        return VerificationResponse(
            verified=False, 
            message="Error: Found zero or multiple faces in the image. Please use a clear photo with one face."
        )
        
    incoming_embedding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    
    # 4. Compare 'embedding' against your Firebase Firestore records
    try:
        members_ref = db.collection("MEMBERS")
        docs = members_ref.stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    best_match = None
    # Tolerance for face match (0.6 is typical for face_recognition/dlib model)
    min_distance = 0.6 

    for doc in docs:
        member_data = doc.to_dict()
        stored_embedding = member_data.get("embedding")
        
        if stored_embedding and len(stored_embedding) == 128:
            distance = calculate_distance(incoming_embedding, stored_embedding)
            
            if distance < min_distance:
                min_distance = distance
                best_match = member_data

    # 5. Return results
    if best_match:
        return VerificationResponse(
            verified=True,
            name=best_match.get("name", "Unknown"),
            club=best_match.get("club", "Unknown"),
            message="Match found."
        )
    else:
        return VerificationResponse(
            verified=False,
            message="Face not recognized."
        )

@app.post("/verify", response_model=VerificationResponse)
async def verify_image(file: UploadFile = File(...)):
    if not db:
        raise HTTPException(status_code=500, detail="Firebase is not configured.")

    # 1. Read the uploaded image file sent from the Flutter app asynchronously
    contents = await file.read()
    
    # Run the CPU-bound face recognition and synchronous Firebase calls in a separate thread
    # This prevents the FastAPI event loop from being blocked, keeping the API fully asynchronous.
    response = await asyncio.to_thread(process_image_and_match, contents)
    
    return response

@app.get("/")
async def root():
    return {"message": "Face Verification API is running. Send POST requests with an image file to /verify."}
