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
	•	Flask 

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
![1](https://github.com/user-attachments/assets/4a7dd795-12e2-4754-83c0-85db05258dfc)

![2](https://github.com/user-attachments/assets/d9974f55-0e95-4665-b089-8a38aaf45bfd)

![3](https://github.com/user-attachments/assets/92feba67-d55d-470a-a9bb-5bdda4ad5616)



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
