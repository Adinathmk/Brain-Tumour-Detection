# Brain_tumour_detection

🧠 Brain Tumour Detection Using Image Segmentation

This project aims to detect and classify brain tumours from MRI scans using image segmentation techniques. A U-Net-based deep learning model is used to segment tumour regions, followed by volumetric analysis and severity classification into five categories.

📌 Features
	•	Upload MRI brain images via a simple web interface
	•	Segment brain tumour regions using a trained U-Net model
	•	Visualize the segmented mask with color-coded overlays
	•	Calculate tumour volume
	•	Classify tumour severity: No Tumour, Low, Moderate, High, and Very High

💻 Technologies Used

🧠 AI/ML
	•	Python
	•	TensorFlow / Keras
	•	U-Net architecture
	•	OpenCV & NumPy for image processing

🌐 Front-End
	•	HTML, CSS, JavaScript
	•	Bootstrap, Font Awesome, Google Fonts
	•	jQuery, WOW.js, Animate.css
	•	Email.js for contact form

📊 Back-End
	•	Flask (optional, if a backend is used for processing)

📁 Dataset

We used publicly available annotated MRI datasets for training and validation. 
https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-dice-coef-89-6/input

🚀 How to Run the Project
	1.	Clone the Repository

git clone [https://github.com/yourusername/brain-tumour-detection.git](https://github.com/Adinathmk/Brain-Tumour-Detection.git)
cd brain-tumour-detection


	2.	Set up a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


	3.	Install dependencies

pip install -r requirements.txt


	4.	Run the application

python app.py



	5.	Open your browser and navigate to http://localhost:5000


📷 Screenshots
![1](https://github.com/user-attachments/assets/370c7ef1-1178-4f0d-9932-ea78c47a4329)

![WhatsApp Image 2025-04-03 at 21 29 39](https://github.com/user-attachments/assets/2e529b61-ffa0-4aed-9466-ddf3dfd9acf0)
![WhatsApp Image 2025-04-03 at 21 29 39 (1)](https://github.com/user-attachments/assets/38260908-664c-4f8d-adf1-79c16a07870c)
![WhatsApp Image 2025-03-16 at 22 29 43 (1)](https://github.com/user-attachments/assets/02741076-4835-438e-bd6a-4361272f91ec)


📈 Results
	•	Achieved accurate tumour segmentation 
	•	Enabled precise volumetric calculation for severity grading
	•	Improved interpretability and user experience through visual outputs

👩‍💻 Team Members
	•	Adinath M K
	•	Anuvinda R
	•	Arshad E


📬 Contact

For queries, feel free to reach out to us at:
📧 adinathmkclt@gmail.com
