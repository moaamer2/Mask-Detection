## Mask Detection Project

### Overview
This project is designed to detect whether individuals are wearing face masks using deep learning and computer vision. The system can be used in real-time applications, such as surveillance systems, to ensure compliance with health and safety protocols.

---

### Features
- Real-time mask detection.
- High accuracy using pre-trained models like MobileNetV2.
- Scalable and deployable as a web application.

---

### Setup Instructions

#### Prerequisites

1. **Hardware Requirements:**
   - NVIDIA GPU with CUDA support ( GTX 1080 TI).
   - At least 8 GB of RAM (16 GB recommended).

2. **Software Requirements:**
   - Python 3.8 or higher.
   - The following libraries:
     - OpenCV
     - TensorFlow/Keras
     - NumPy
     - Matplotlib
     - Sklearn

#### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/moaamer2/Mask-Detection.git
   
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**
   - Obtain the dataset from Kaggle or another reliable source.
   - Place the dataset in the `data/` directory.

5. **Run Preprocessing:**
   - Resize, normalize, and augment images by running the preprocessing script:
     ```bash
     python preprocess_data.py
     ```

6. **Train the Model:**
   - Train the model using the training script.    
   - The trained model will be saved as `mask_detection_best.h5` in the `models/` directory.

7. **Test the Model:**
   - Evaluate the model on the test dataset.

8. **Run the Application:**
   ```bash
   app.py
   ```

---

### Directory Structure

```plaintext
mask-detection/
├── data/
│   └── New_Masks_Dataset/
│       ├── Train/
│       └── Test/
├── models/
│   └── mask_detection_best.h5
├── scripts/
│   ├── preprocess_data.py
│   └── train_model.py
├── app.py
└── README.md
```

---

### Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

