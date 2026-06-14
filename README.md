# Face Verification API

A high-performance FastAPI backend designed to process facial images, extract 128-dimensional facial embeddings using `face_recognition`, and securely verify them against member records stored in Firebase Firestore. 

Designed specifically as a remote backend for mobile clients (like Flutter), allowing heavy AI processing to be offloaded from the client device.

## Features
- **File Upload Endpoint**: Accepts raw image files (`.jpg`, `.png`).
- **Heavy AI Offloading**: Uses OpenCV and dlib-based `face_recognition` to calculate embeddings.
- **Asynchronous Execution**: Uses `asyncio.to_thread()` to prevent heavy CPU and database operations from blocking the API event loop.
- **Firebase Integration**: Securely connects to Firestore to retrieve existing member embeddings.
- **CORS Configured**: Fully configured to accept cross-origin requests.

---

## Architecture Overview

1. **Flutter App** takes a photo and POSTs it as `multipart/form-data` to `/verify`.
2. **FastAPI** decodes the image bytes into an OpenCV matrix.
3. **Face Recognition** locates the face and extracts a mathematical embedding (128 floats).
4. **Firebase Admin** fetches all registered members from the `MEMBERS` collection in Firestore.
5. **Matching Engine** calculates the Euclidean distance between the incoming embedding and the stored embeddings. If the distance is `< 0.6`, it's considered a match.
6. The API returns the member's `name`, `club`, and verification status.

---

## Deployment to Hugging Face Spaces

This API is Docker-ready and can be deployed directly to Hugging Face Spaces.

### 1. Update/Upload Files
You need to commit the following 3 files to your Hugging Face Space repository:
- `api.py` (The main application)
- `requirements-api.txt` (The Python dependencies)
- `Dockerfile` (The container instructions)

*Whenever you make changes to `api.py`, simply upload the new `api.py` file to your Hugging Face repository. Hugging Face will automatically detect the change and rebuild the Docker container.*

### 2. Configure Firebase Secrets
Do **not** upload your `serviceAccountKey.json` directly to Hugging Face.
1. Go to your Hugging Face Space.
2. Click on **Settings** > **Variables and secrets**.
3. Under **Secrets**, create a new secret named `FIREBASE_CREDENTIALS`.
4. Open your `serviceAccountKey.json` file in a text editor, copy all of the contents, and paste it into the value field. 
5. Save the secret. 

The API is coded to automatically read this secret when it starts up.

---

## Local Development

If you want to run and test the API on your own machine before pushing to Hugging Face:

1. **Install Dependencies**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Add Credentials**
   Place your `serviceAccountKey.json` in the root folder (same folder as `api.py`). The API will detect it automatically if the `FIREBASE_CREDENTIALS` environment variable isn't set.

3. **Start the Server**
   ```bash
   uvicorn api:app --reload
   ```

4. **Test the API**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the interactive Swagger UI where you can directly upload a photo to test the verification!

---

## API Endpoints

### `POST /verify`
Verifies an uploaded face image against the database.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (The image file)

**Successful Response (200 OK):**
```json
{
  "verified": true,
  "name": "Jane Doe",
  "club": "Engineering",
  "message": "Match found."
}
```

**Unsuccessful Response (200 OK):**
```json
{
  "verified": false,
  "name": null,
  "club": null,
  "message": "Face not recognized."
}
```

**Error Response (Face not found / Multiple faces):**
```json
{
  "verified": false,
  "name": null,
  "club": null,
  "message": "Error: Found zero or multiple faces in the image. Please use a clear photo with one face."
}
```
