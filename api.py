from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from typing import List, Optional

# Initialize FastAPI
app = FastAPI(title="Face Verification API", description="API to verify face embeddings against Firebase Firestore")

# Initialize Firebase (Make sure serviceAccountKey.json is present in the root folder)
# In production, use environment variables for credentials.
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Warning: Firebase initialization failed. Ensure serviceAccountKey.json exists. Error: {e}")
    db = None

class VerificationRequest(BaseModel):
    embedding: List[float]

class VerificationResponse(BaseModel):
    verified: bool
    name: Optional[str] = None
    club: Optional[str] = None
    message: str

def calculate_distance(embedding1, embedding2):
    """Calculate Euclidean distance between two face embeddings."""
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

@app.post("/verify", response_model=VerificationResponse)
async def verify_face(request: VerificationRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Firebase is not configured.")

    if len(request.embedding) != 128:
        raise HTTPException(status_code=400, detail="Embedding must be a list of 128 floats.")

    incoming_embedding = np.array(request.embedding)
    
    # 1. Fetch members from Firebase
    try:
        members_ref = db.collection("MEMBERS")
        docs = members_ref.stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    best_match = None
    # Tolerance for face match (0.6 is typical for face_recognition/dlib model)
    min_distance = 0.6 

    # 2. Compare embeddings
    for doc in docs:
        member_data = doc.to_dict()
        stored_embedding = member_data.get("embedding")
        
        if stored_embedding and len(stored_embedding) == 128:
            distance = calculate_distance(incoming_embedding, stored_embedding)
            
            if distance < min_distance:
                min_distance = distance
                best_match = member_data

    # 3. Return results
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
            message="No match found."
        )

@app.get("/")
async def root():
    return {"message": "Face Verification API is running. Send POST requests to /verify."}
