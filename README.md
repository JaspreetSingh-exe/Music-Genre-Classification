# **🎵 Music Genre Classification 🎶**
A deep learning-based system to classify music genres using **SVM, CNN, LSTM, and Transformer models**, with **YAMNet-based feature extraction**.  

## **🚀 Features**
✅ **Extracts audio features** using **YAMNet embeddings**.  
✅ **Trains multiple models** (**SVM, CNN, LSTM, Transformer**) for comparative analysis.  
✅ **Evaluates model performance** using accuracy, confusion matrices, and classification reports.  
✅ **Allows users to upload an audio file** for real-time genre classification.  

---

## **🎵 What is YAMNet?**
YAMNet is a **pre-trained deep learning model developed by Google** for **audio classification**. It is based on **MobileNet architecture** and is trained on **AudioSet**, a large-scale dataset of sounds. YAMNet extracts **high-level embeddings** from audio, which can be used for various machine learning tasks, including **music genre classification**.

### **🎯 Why YAMNet?**
- 🟢 **Pre-trained on large-scale data** (AudioSet)
- 🟢 **Extracts meaningful sound features**
- 🟢 **Lightweight and efficient for real-time applications**

### **📌 Using YAMNet for Feature Extraction**
To extract features from an audio file using YAMNet in Python:

```python
import tensorflow_hub as hub
import numpy as np
import librosa

# Load the YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load and preprocess the audio file
def extract_yamnet_features(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)  # Ensure 16kHz sample rate
    waveform = np.reshape(waveform, (1, -1))  # Reshape to match model input
    
    # Run YAMNet model
    embeddings, scores, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

# Example usage
features = extract_yamnet_features('test_music(hiphop).wav')
print(features.shape)  # Output shape: (number of frames, 1024)
```

🔗 **Reference:** [TensorFlow YAMNet Tutorial](https://www.tensorflow.org/hub/tutorials/yamnet)

---

## **📂 Project Structure**
```
🌆 Music Genre Classification
 ├── 📂 Data
 │   ├── 📂 genres_original        # Raw music dataset organized by genre
 │   ├── 📂 images_original        # Spectrogram images (if applicable)
 │   ├── 📝 music_features.csv     # Extracted features dataset
 │   ├── 📝 test_music(hiphop).wav # Sample test audio file
 │   ├── 📝 yamnet_features.csv    # YAMNet feature embeddings
 │
 ├── 📂 Models
 │   ├── 📝 cnn_model.h5           # Trained CNN model
 │   ├── 📝 lstm_model.h5          # Trained LSTM model
 │   ├── 📝 svm_model.pkl          # Trained SVM model
 │   ├── 📝 transformer_model.h5   # Trained Transformer model
 │
 ├── 📝 .gitattributes             # Git LFS tracking for large files
 ├── 📝 requirements.txt           # Required dependencies
 ├── 📝 FeatureExtraction.ipynb    # Extracts YAMNet features from audio
 ├── 📝 MusicGenreModels.ipynb     # Trains all ML models (SVM, CNN, LSTM, Transformer)
 ├── 📝 Result_Analysis.ipynb      # Evaluates model performance & visualization
```

---

## **🛠️ Setup & Installation**
### **1⃣ Clone the Repository**
```sh
git clone https://github.com/JaspreetSingh-exe/Music-Genre-Classification.git
cd Music-Genre-Classification
```

### **2⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3⃣ Download & Install Git LFS (For Large Model Files)**
```sh
git lfs install
git lfs pull
```

---

## **🎵 How to Run the Project**
### **1⃣ Extract Features from Audio**
Run:
```sh
jupyter notebook FeatureExtraction.ipynb
```
This extracts **YAMNet embeddings** from the dataset.

### **2⃣ Train Models**
Run:
```sh
jupyter notebook MusicGenreModels.ipynb
```
This trains **SVM, CNN, LSTM, Transformer** models and saves them in the `Models/` directory.

### **3⃣ Evaluate Model Performance**
Run:
```sh
jupyter notebook Result_Analysis.ipynb
```
This generates **accuracy scores, confusion matrices, and classification reports**.

### **4⃣ Predict Genre for a New Audio File**
Run in Python:
```python
from MusicGenreModels import predict_audio_file
predict_audio_file("path/to/audio.wav", models, scaler, label_encoder)
```

---

## **📊 Model Comparison**
| Model         | Accuracy |
|--------------|----------|
| **SVM**        | 86.11% (best) |
| **CNN**        | 80.56% (overfitting) |
| **LSTM**       | 71.67% |
| **Transformer** | 33.33% |


---

## **📌 Future Improvements**
- ✅ Deploy as a **Web App** where users can upload songs & get genre predictions.
- ✅ Improve accuracy using **larger datasets** & **better preprocessing**.
- ✅ Experiment with **other pre-trained models** for better feature extraction.

---

## **🐜 License**
This project is licensed under the **MIT License**.

---

## **📞 Contact**
💡 **Author:** Jaspreet Singh  
📞 **Email:** jaspreetsingh01110@gmail.com   
🔗 **GitHub:** [JaspreetSingh-exe](https://github.com/JaspreetSingh-exe)  

---

