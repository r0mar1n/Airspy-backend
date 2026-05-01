AQI Backend Setup Instructions

1. Install Python (3.9+)

2. Open terminal in this folder

3. Create virtual environment:
   python -m venv venv

4. Activate it:

   Windows:
   venv\Scripts\activate

   Mac/Linux:
   source venv/bin/activate

5. Install dependencies:
   pip install -r requirements.txt

6. Run the API:
   uvicorn main:app --reload

7. Open:
   http://127.0.0.1:8000/docs

NOTE:

* Keep the "models" folder in the same directory
