import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore
import sys

def add_member(image_path, member_name, club_name):
    print(f"Initializing Firebase...")
    cred = credentials.Certificate("serviceAccountKey.json")
    # Check if already initialized (in case run multiple times in REPL)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    print(f"Loading image {image_path}...")
    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Detecting faces...")
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        print("Error: No face found in the image.")
        return
    elif len(face_locations) > 1:
        print("Error: Multiple faces found. Please use a photo with only one person.")
        return
        
    print("Extracting embedding...")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    embedding = face_encodings[0].tolist() 
    
    print(f"Uploading {member_name} to Firebase...")
    doc_ref = db.collection("MEMBERS").document()
    doc_ref.set({
        "name": member_name,
        "club": club_name,
        "embedding": embedding
    })
    
    print("✅ Success: Member successfully added to Firebase!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python add_member_to_firebase.py <image_path> <name> <club>")
        print("Example: python add_member_to_firebase.py Yashodeep.jpeg \"Yashodeep\" \"Tech Club\"")
        sys.exit(1)
        
    add_member(sys.argv[1], sys.argv[2], sys.argv[3])
