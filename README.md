# FINGERPRINT-X
### Experimental Fingerprint Image Classification with Deep Learning
FINGERPRINT-X is an experimental computer vision project that uses TensorFlow/Keras to classify fingerprint images into two dataset labels: Female and Male.
The project includes model training, test-set evaluation, image preprocessing, and a Streamlit interface for experimental inference.
> **Disclaimer:** This is an experimental machine-learning project, not a validated biometric or gender-assessment system. Predictions may be inaccurate and should not be used for identity verification, employment, medical decisions, or other high-impact decisions.
## Features
- Fingerprint image classification using a CNN
- Image preprocessing with grayscale conversion and resizing
- Model training using TensorFlow/Keras
- Evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Streamlit-based interactive interface
- Separate training, evaluation, and application scripts
## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- OpenCV
- Streamlit
- Pillow
## Project Structure
```text
FINGERPRINT-X/
├── app.py
├── main.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── architecture.png
├── fingerprints/
│   └── process.py
├── dataset/                   # Local dataset; not tracked by Git
├── fingerprint_gender_model_v2.keras  # Generated model; not tracked by Git
└── README.md
```
## Setup
### 1. Clone the repository
```bash
git clone https://github.com/yash01100/FINGERPRINT-X.git
cd FINGERPRINT-X
```
### 2. Create and activate a virtual environment
Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Dataset
The dataset is not included in this repository. Place the dataset in the expected local directory structure before training or evaluation.
The current evaluation script expects test images under:
```text
dataset/
└── test/
```
Use only datasets you have permission to access and process. Follow the dataset provider's licence and usage conditions.
## Train the Model
Activate the virtual environment, make sure your dataset is prepared, then run:
```bash
python train_model.py
```
The training script is configured to save the V2 model as:
```text
fingerprint_gender_model_v2.keras
```
Training time and results depend on your hardware, dataset, and configuration.
## Evaluate the Model
Run:
```bash
python evaluate_model.py
```
The evaluation script uses the configured test dataset and reports classification metrics and a confusion matrix.
### Recorded Test Results
The following results were recorded from a run on 900 test images:
| Metric | Result |
|---|---:|
| Test accuracy | 71.11% |
| Female precision | 0.46 |
| Female recall | 0.70 |
| Female F1-score | 0.55 |
| Male precision | 0.87 |
| Male recall | 0.71 |
| Male F1-score | 0.79 |
| Test images | 900 |
Confusion matrix from that run:
```text
                 Predicted Female   Predicted Male
Actual Female          161                 69
Actual Male            191                479
```
These are recorded experimental results, not a guarantee of future performance. The test set contains 230 Female-labelled images and 670 Male-labelled images, so class imbalance is important when interpreting accuracy. Results alone do not establish real-world reliability or scientific validity.
## Run the Streamlit App
Ensure that the trained V2 model file is available locally, then run:
```bash
streamlit run app.py
```
The app uses the model path configured in `app.py`. The model file is excluded from Git, so it must be generated locally or obtained through an authorised distribution method.
## Limitations
- Predictions can be incorrect.
- The test data has unequal class counts.
- Performance on other datasets or fingerprint capture conditions has not been established.
- The model's output should not be treated as a reliable determination of a person's gender.
- Dataset permissions, privacy, and consent must be respected.
## Future Improvements
- Improve evaluation across balanced and independent datasets.
- Add reproducible training configurations and experiment tracking.
- Document preprocessing and model architecture in detail.
- Add automated tests and input validation.
- Publish model files only with appropriate documentation and licensing.
## Author
**Yash Raj**
GitHub: [@yash01100](https://github.com/yash01100)
---
This project is intended for educational and research experimentation.
