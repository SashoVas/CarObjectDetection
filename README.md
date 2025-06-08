# CarObjectDetection

## How to Start the App

### 1. Using Docker Compose (Recommended)

This project uses Docker Compose to run both the backend (FastAPI) and frontend (Angular) services.

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed
- [Docker Compose](https://docs.docker.com/compose/install/) (if not included with Docker Desktop)

#### Steps
1. Open a terminal in the project root directory (where `docker-compose.yml` is located).
2. Run:
   ```sh
   docker compose up --build
   ```
3. The frontend will be available at [http://localhost:4200](http://localhost:4200)
4. The backend API will be available at [http://localhost:8000](http://localhost:8000)

---

### 2. Manual Start (Development Mode)

#### Prerequisites
- [Node.js](https://nodejs.org/) (v18 or later recommended)
- [npm](https://www.npmjs.com/) (comes with Node.js)
- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

#### Backend (FastAPI)
1. Open a terminal and navigate to the `backend` directory:
   ```sh
   cd backend
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```sh
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Or
   ```sh
   fastapi dev main.py
   ```
   The backend will be running at [http://localhost:8000](http://localhost:8000)

#### Frontend (Angular)
1. Open a new terminal and navigate to the `frontend` directory:
   ```sh
   cd frontend
   ```
2. Install Node.js dependencies:
   ```sh
   npm install
   ```
3. Start the Angular development server:
   ```sh
   ng serve --open
   ```
   The frontend will be running at [http://localhost:4200](http://localhost:4200)

---

### Notes
- The backend will look for YOLO model files in the `backend/models` directory. At least one `.pt` model file should be present for detection to work.
- The frontend expects the backend to be running at `http://localhost:8000`.