# Sign Language Recognition Project

This project implements a real-time hand sign language recognition system using computer vision and machine learning. It can detect hand gestures, classify them into alphabets, and convert the detected signs into meaningful sentences using AI.

## 🚀 Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **Sign Classification**: Machine learning model trained on custom hand gesture dataset
- **Quality Control**: Visual feedback during data collection with hand detection validation
- **AI-Powered Sentence Generation**: Converts detected alphabets into coherent sentences using Google's Gemini AI

## 📋 Prerequisites

- Python 3.11
- Webcam for real-time detection
- Google API key for sentence generation

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Priyanshu-pulak/Sign-Language
cd Sign-Language
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate signLanguage
```

3. **Set up environment variables:**
Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## 🎯 Usage

### 1. Data Collection
Collect hand gesture images for training:
```bash
python collect_imgs.py
```
- Shows real-time hand detection feedback
- Press 'S' to start capturing for each alphabet
- Captures 200 images per class with quality control

### 2. Dataset Creation
Process collected images and create training dataset:
```bash
python create_dataset.py
```
- Extracts hand landmarks using MediaPipe
- Normalizes landmarks relative to wrist position
- Includes data augmentation with rotation

### 3. Model Training
Train the classification model:
```bash
python train_classifier.py
```
- Uses Support Vector Machine (SVM) classifier
- Displays training accuracy
- Saves trained model as `model.pkl`

### 4. Real-time Inference
Run real-time sign language detection:
```bash
python inference_classifier.py
```
- Press 'Q' to quit
- Detected alphabets are saved to `detected_alphabets.txt`

### 5. Generate Sentences
Convert detected alphabets to meaningful sentences:
```bash
python textapi.py
```
- Uses Google's Gemini AI for natural language processing
- Outputs generated sentence to `convert_sentence.txt`

### 6. Text-to-Speech (Optional)
Convert generated text to speech:
```bash
python speechapi.py
```

## 📁 Project Structure

```
Sign-Language/
├── collect_imgs.py          # Data collection with hand detection
├── create_dataset.py        # Dataset creation and preprocessing
├── train_classifier.py     # Model training
├── inference_classifier.py # Real-time inference
├── textapi.py              # AI sentence generation
├── speechapi.py            # Text-to-speech conversion
├── app.py                  # Streamlit web application
├── environment.yml         # Conda environment configuration
├── .env                    # Environment variables (create this)
├── Data/                   # Training images (auto-created)
├── model.pkl              # Trained model (generated)
├── data.pkl               # Processed dataset (generated)
└── README.md              # This file
```

## 🔧 Key Features Explained

### Hand Detection with Quality Control
- Real-time visualization of hand landmarks during data collection
- Only captures images when hands are properly detected
- Ensures high-quality training data

### Landmark Normalization
- All hand landmarks are normalized relative to wrist position
- Improves model robustness across different hand sizes and positions

### Data Augmentation
- Includes rotation augmentation to improve model generalization
- Triples the effective dataset size

### AI Integration
- Uses Google's Gemini AI for intelligent sentence formation
- Converts raw alphabet sequences into meaningful text

## 👨‍💻 Author

**Priyanshu Pulak**
- GitHub: [@Priyanshu-pulak](https://github.com/Priyanshu-pulak)
- Repository: [Sign-Language](https://github.com/Priyanshu-pulak/Sign-Language)

## 🙏 Acknowledgments

- MediaPipe for hand detection capabilities
- Google's Gemini AI for natural language processing
- scikit-learn for machine learning algorithms
- OpenCV for computer vision functionality
