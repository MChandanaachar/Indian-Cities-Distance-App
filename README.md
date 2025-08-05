🚗 Indian Cities Distance Prediction Project
This project aims to analyze and predict distances between Indian cities using a machine learning approach, and also provides a simple web-based interface to look up distances using a Streamlit app.

📁 Project Structure
bash
Copy
Edit
ml-project/
│
├── read_data.py          # ML analysis and model
├── distance.py           # Streamlit app
├── .gitignore            # Files/folders ignored by Git
├── requirements.txt      # Project dependencies
└── indian-cities-dataset.csv  # Dataset file
📌 Features
Load and preprocess Indian cities distance dataset.

Perform basic visualizations using Matplotlib and Seaborn.

Train a Linear Regression model to predict distances.

Normalize and encode categorical data.

Deploy a simple distance lookup app using Streamlit.

🔧 Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/MChandanaachar/Indian-Cities-Distance-App.git
cd your-repo-name
Create and activate a virtual environment

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate    # For Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
🧪 Run the ML Script
To run the data analysis and machine learning pipeline:

bash
Copy
Edit
python read_data.py
🌐 Run the Streamlit App
To launch the Streamlit web app:

bash
Copy
Edit
streamlit run distance.py
It will open in your default browser. You can select cities from dropdowns to view the distance between them.

🗂️ Files Explained
read_data.py
Loads data, processes it, trains a model, and evaluates performance.

distance.py
Streamlit app for distance lookup between city pairs.

indian-cities-dataset.csv
Dataset containing city pairs and distances.

📦 Example .gitignore
gitignore
Copy
Edit
venv/
__pycache__/
*.pyc
*.log
.env
✅ Requirements
Your requirements.txt can include:

nginx
Copy
Edit
pandas
matplotlib
seaborn
scikit-learn
streamlit
You can generate it using:

bash
Copy
Edit
pip freeze > requirements.txt
✅ Future Ideas
Add map visualizations using Folium or Plotly.

Integrate the trained model into the Streamlit app.

Accept user input for prediction instead of hardcoded data.
