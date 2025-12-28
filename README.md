# 🎭 Real-Time Facial Emotion Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.8.0-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Truth From Facial Expressions With Deep Learning**  
> A CNN-based deep learning system for real-time emotion detection from facial expressions

**Author:** Kunj Mungalpara  
**Project:** SEM 4 SGP-2

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo & Screenshots](#demo--screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Applications](#applications)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a real-time facial emotion detection system using Convolutional Neural Networks (CNN) and deep learning techniques. The system can analyze facial expressions and classify them into seven different emotion categories with high accuracy.

The project uses computer vision algorithms for face detection and deep learning models for emotion classification, providing instant emotional analysis from webcam feeds or uploaded images.

### 🌟 Key Highlights
- **Real-time emotion detection** from webcam or video streams
- **Multiple interfaces**: Streamlit web app and Gradio UI
- **Music playlist recommendation** based on detected emotions
- **High accuracy** model trained on 35,887 facial expression images
- **Seven emotion categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## ✨ Features

### 🎥 Multiple Detection Modes
- **Webcam Mode**: Real-time emotion detection from live camera feed
- **Image Upload**: Analyze emotions from uploaded images
- **Batch Processing**: Aggregate emotion predictions over multiple frames

### 🎨 User Interfaces
- **Streamlit App** (`app.py`): Clean, modern web interface with image upload and webcam support
- **Gradio Interface** (`gradioo.py`): Interactive UI with tabs for webcam and image analysis
- **OpenCV Demo** (`main.py`): Lightweight demonstration with playlist integration

### 🎵 Smart Features
- **Emotion-based Music Recommendations**: Automatically suggests YouTube Music playlists based on detected mood
- **Aggregated Predictions**: Analyzes last 20 frames to determine the most frequent emotion
- **Face Detection**: Uses Haar Cascade classifier for accurate face localization

---

## 📸 Demo & Screenshots

### System in Action
![Emotion Detection Demo 1](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/f560dbaa-2adb-4e6d-8541-1549dc06d64b)
![Emotion Detection Demo 2](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/b22230dd-e68c-4ba4-b43c-efc1a7a48660)

### Real-time Analysis
![Real-time Detection 1](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/d885fb4f-0ef9-4439-9b6f-a485408b4ab2)
![Real-time Detection 2](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/4787e6f7-1c60-4fea-ace5-f6d5e4d04ebb)

### Interface Screenshots
![Streamlit Interface](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/0e1a1911-2a6c-4e11-8672-b925a7f60399)
![Results Visualization 1](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/bcb8dac7-2ddc-4f8c-b789-d985dc287257)
![Results Visualization 2](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/8492315d-48a1-479f-977e-477ad0860de6)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/kunj24/facebeats.git
cd facebeats
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## 💻 Usage

### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Features:**
- Upload images for emotion analysis
- Enable webcam for real-time detection
- Aggregates last 20 frames for stable predictions

### Option 2: Gradio Interface
```bash
python gradioo.py
```
Access the interface at `http://localhost:7860`

**Features:**
- Separate tabs for webcam and image upload
- Interactive controls
- Automatic music playlist recommendations

### Option 3: OpenCV Demo
```bash
python main.py
```

**Features:**
- Minimal setup with direct OpenCV display
- Fixed capture duration with emotion aggregation
- Automatic playlist redirection based on mood

---

## 📊 Dataset

### Face Expression Recognition Dataset
- **Total Images**: 35,887 grayscale facial images
- **Training Set**: 28,821 images
- **Test Set**: 7,066 images (Public + Private)
- **Image Size**: 48×48 pixels
- **Format**: Grayscale

### Emotion Distribution
| Emotion | Count | Percentage |
|---------|-------|------------|
| Happy | 7,164 | 20.0% |
| Sad | 4,938 | 13.8% |
| Fear | 4,103 | 11.4% |
| Neutral | 4,982 | 13.9% |
| Angry | 3,993 | 11.1% |
| Surprise | 3,205 | 8.9% |
| Disgust | 436 | 1.2% |

---

## 🧠 Model Architecture

### Convolutional Neural Network (CNN)
The emotion classification model uses a deep CNN architecture optimized for facial expression recognition:

- **Input Layer**: 48×48 grayscale images
- **Convolutional Layers**: Multiple conv layers with ReLU activation
- **Pooling Layers**: Max pooling for feature reduction
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: 7 neurons with softmax activation (7 emotion classes)

### Model Performance
- **Training Accuracy**: ~78.9%
- **Validation Accuracy**: High performance on test set
- **Real-time Processing**: Fast inference for live video streams

### Key Improvements
- Additional layers for enhanced accuracy
- Optimal threshold tuning
- Tested with commonly used combinations from recent research
- Superior performance compared to traditional methods

---

## 🎯 Applications

### 🏢 Commercial Applications
- **Marketing & Advertising**: Analyze customer emotional reactions to products/campaigns
- **Customer Experience**: Monitor and improve service quality based on emotional feedback
- **Retail Analytics**: Track shopper emotions for store optimization

### 🏥 Healthcare
- **Mental Health Monitoring**: Early detection of depression, anxiety, and mood disorders
- **Patient Care**: Assess patient comfort and pain levels
- **Alzheimer's & Schizophrenia**: Early detection through facial expression analysis

### 🎓 Education
- **Student Engagement**: Monitor student attention and understanding in classrooms
- **E-Learning Platforms**: Adapt content based on learner emotional state
- **Special Education**: Support autism diagnosis and intervention

### 🚗 Safety & Security
- **Driver Monitoring**: Detect drowsiness and distraction in autonomous vehicles
- **Security Systems**: Enhance surveillance with emotion-based threat detection
- **Border Control**: Support identity verification processes

### 🎮 Entertainment & Gaming
- **Game Development**: Adapt gameplay based on player emotions
- **Content Recommendation**: Suggest content matching user mood
- **Virtual Reality**: Enhance immersive experiences with emotion tracking

### 🔬 Research & Academia
- **Psychology Research**: Study emotional responses in controlled experiments
- **Human-Computer Interaction**: Improve interface design based on user emotions
- **Social Science**: Analyze collective emotional responses to events

---

## 📈 Results

### Performance Metrics
- Successfully classifies 7 distinct emotional states
- Real-time processing with minimal latency
- Robust face detection using Haar Cascade classifier
- Stable predictions through frame aggregation

### Research Background
This project builds upon established research in emotion detection:
- **Pang and Lee**: Pioneered machine learning methods with 78.9% success rate
- **Dandil & Özdemir (2019)**: Real-time emotion recognition using AlexNet CNN
- **Chen (2015)**: ESA-CNN for automatic feature learning

### Competitive Advantages
✅ Higher accuracy through optimized layer combinations  
✅ Real-time response with instantaneous feedback  
✅ Enhanced readability of emotional expressions  
✅ Fast model restructuring and improvement capabilities  

---

## 🛠️ Project Structure

```
real_time_facial_emotion_detection/
│
├── app.py                          # Streamlit web interface
├── gradioo.py                      # Gradio interactive UI
├── main.py                         # OpenCV minimal demo
├── graph.py                        # Visualization utilities
├── model.h5                        # Trained CNN model
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── emotion-classification-cnn-using-keras.ipynb  # Model training notebook

```

---

## 📦 Dependencies

### Core Libraries
- **TensorFlow** (2.18.0): Deep learning framework
- **Keras** (3.8.0): High-level neural networks API
- **OpenCV** (cv2): Computer vision and image processing
- **NumPy**: Numerical computations
- **Streamlit**: Web app framework
- **Gradio**: Interactive ML interfaces

### Additional Requirements
- absl-py, grpcio, protobuf (TensorFlow dependencies)
- h5py (Model file handling)
- Pillow (Image processing)

See [requirements.txt](requirements.txt) for complete list.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contribution
- Add more emotion categories
- Improve model accuracy
- Optimize real-time performance
- Add new interface options
- Expand music playlist database
- Implement mobile app version

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Face Expression Recognition Dataset creators
- TensorFlow and Keras communities
- OpenCV contributors
- Research papers that inspired this work:
  - Pang and Lee (Machine Learning for Emotion Analysis)
  - Dandil & Özdemir (2019) - AlexNet CNN for Emotion Recognition
  - Chen (2015) - ESA-CNN Architecture

---

## 📧 Contact

**Kunj Mungalpara**

- GitHub: [@kunj24](https://github.com/kunj24)
- Project Link: [FaceBeats](https://github.com/kunj24/facebeats)

---

## 🔄 Recent Updates (2025)

### ✨ Latest Changes
- **Cleaned up repository**: Removed temporary files and cache directories
- **Enhanced documentation**: Complete rewrite of README with modern formatting
- **Deprecated legacy scripts**: Streamlined to three main entry points
- **Improved structure**: Better organization of project files

### 🎯 Active Entry Points
- `app.py`: Streamlit interface (image upload + optional webcam)
- `gradioo.py`: Gradio interface (tabs for webcam and image)
- `main.py`: Minimal OpenCV demo with playlist redirection

---

## 📚 Further Reading

### Similar Platforms & Services
- **Affectiva (Affdex)**: MIT-developed emotion AI platform
- **Microsoft Azure Cognitive Services**: Face API for emotion detection
- **Amazon Rekognition**: Facial analysis and emotion detection
- **Face++**: Computer vision and facial recognition platform

### Research Areas
- Computer Vision for Facial Analysis
- Deep Learning in Emotion Recognition
- Human-Computer Interaction
- Affective Computing
- Convolutional Neural Networks for Image Classification

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ by Kunj Mungalpara

</div>

**1.2 Similar Examples in the World and in Turkey**
<br>
1.2 Similar Examples in the World and in Turkey
Real-time emotion detection has attracted a lot of attention in the field of artificial intelligence and image processing in recent years. For this reason, there are many similar examples in the world and in our country. When similar projects that can be accessed as a result of the research are examined, it is found that models that perform face recognition, gender classification and emotion classification simultaneously in a single step using real-time convolutional neural network architecture have created awareness. Applications that predict age based on facial expressions have been developed and large data sets have been used for these applications. It is known that platforms such as "Affdex" developed by MIT, Microsoft's "Azure Cognitive Services" and "Face API, Amazon "Rekognition", "SkyBiometry", Emotion AI", "Face++", "Anthropic" are working on real-time emotion detection projects. These projects use deep learning techniques to classify emotional responses by building sentiment analysis models using large datasets. Pang and Lee pioneered the machine learning method, which has the most effective results in the studies conducted in the field of emotion analysis in the literature. The best success rate was 78.9%. 
In Pang and Lee's project, text classification was treated differently. Dandil and 
Özdemir, 2019, in their study, the classical convolutional neural network AlexNet and real 
proposed an emotion recognition system based on facial expressions from timed video frames. 
Chen and Ark, in their 2015 study, proposed an emotion classification system based on
with the automatic learning convolutional neural network (ESA-CNN) that they use to manually 
outperformed traditional methods based on prepared features.
Studies have shown that ESA can be used for feature extraction in emotion recognition. 
shows. The input layer on the convolution structure is the input of the data of the model to be trained. 
is the layer that is transformed into the desired structure. 
Sentiment analysis methods are divided into machine learning-based methods and dictionary-based methods. 
are classified. Among these methods, considering the studies in the literature 
machine learning based methods were found to give maximum accuracy. 
There are some features that distinguish this project from other applications. Detailed 
the fact that the success rates obtained as a result of research cannot exceed a certain limit
and it is noteworthy that most projects do not come close to this accuracy limit. This problem 
In order to provide a solution, additional layers have been added to the model creation part and model training 
commonly used combinations have been tested in recent research to determine certain optimal 
thresholds were tried to be determined. As a result, the evaluation and success rate
has been increased. In addition, compared to other projects, we have developed a system that can return an instantaneous response value. 
software has been developed and the expressions are easier to read. In this way, the model is built in a fast 
and restructured and allowed to be improved.


<br>

**2. PROJECT CONTENT AND SCOPE** 
<br>
Recently, face recognition and detection systems have been used in many commercial, military, security, social and 
is frequently used in applications in psychological fields. Analyses performed; human 
It involves the identification and interpretation of the movements of their faces. By humans 
emotional expressions, which are difficult to analyze even in a computerized environment, can be tested and determined. 
The thought that it will provide convenience has brought popularity to the field of deep learning. This 
computer vision should also be mentioned in this context. Computer vision is nowadays used for face and 
is widely used in the fields of emotion classification. Face recognition, image or 
automatic identification or verification of persons in data derived from videos
process. There are four basic stages of face recognition. These processes are face
detection, normalization, feature extraction and classification. Normalization and 
no matter how successful classification algorithms are in face recognition, if feature extraction 
If this stage is not successful, the system does not achieve the desired success.
The real-time emotion detection project is a combination of image processing and artificial intelligence algorithms. 
consists of a series of steps. The first step is to detect human faces in the snapshot. 
face recognition algorithms were used. Once the face or faces are detected, the emotion 
sentiment analysis model developed using convolutional neural networks to determine the state of 
is used. This model was trained with deep learning techniques and was applied to different emotion categories. 
with the allocated dataset. Finally, the identified emotional states were analyzed 
and present the results to the user.
As a result of the additions made throughout the project scope, the success rate of the revamped model and 
with the help of the graphs drawn that the availability increases in direct proportion. 
observed. One of the deep learning techniques for feature extraction, artificial neural 
a new model using Convolutional Neural Networks (ESA-CNN), an approach involving networks 
has been developed. 
Commonly used combinations for model training have been tested in recent studies and 
The effect of the classification algorithms on the performance of the classification algorithms is analyzed. 
By making evaluations, the classification algorithm with strong performance and the real 
emotion classification from facial images using a time convolutional neural network architecture 
The project that realizes the process simultaneously has been revealed.
<br>
<br>

**3.2 Infrastructure, Hardware and Software Specifications**
A high quality dataset is needed for high performance of deep learning. This 
Therefore, we should look for datasets where training and testing performance is measured to be high and 
the appropriate set should be selected. In the project, the face expression recognition dataset is used for deep 
A project has been developed for emotion recognition with learning. Face expression recognition data 
set fulfills the need in the emotion detection project. Face expression recognition data 
There are a total of 35887 images in the set. 28821 of the images are training and 7066 of the images 
is reserved for Public and Private tests. Public tests are the performance after the model is finished. 
ratio, while Private tests are used to test a portion of the images in the dataset. 
"PrivateTest" and is then used for testing. Used for the project 
The technical details of the images can be analyzed through this data set. Thus, in columns 
can be seen how many groups the samples in the data set used are divided into, and the fields within the set can be called data 
visualization can be applied. The data set used consists of 35887 rows and 3 columns 
consists of. In this data set, there are pictures to identify seven emotions. This 
emotions angry (3993), disgust (436), fear (4103), happy (7164), sad 
(4938), surprise (3205), neutral (4982). The structure of the images is analyzed with the help of functions 
It is 48x48 in size and arranged in shades of gray. The model contains 
The new model was tested by training with the images separately. With the model 
In the study, seven different emotion classes (fear, anger, disgust, 
happiness, neutral, sadness, surprise) were discussed.

<br>
<br>

**4.2 Areas of Use of the System-Software**
Facial expressions are the clearest way for people to express their inner feelings in their daily communication. 
is one of the indicators. A person's physical or mental state can be determined by analyzing their facial expressions. 
can be detected. Facial expression recognition can therefore be used in autopilot, human-computer interaction, medical treatment 
and other areas related to facial expressions. This field is becoming increasingly important 
is becoming a research topic as a result of the researches conducted. Face
psychological science, such as emotion recognition from expression, identification of autism and schizophrenia, 
assessments, detection of a drowsy driver, Alzheimer's disease or schizophrenia 
identification of abnormalities in the early stages, medical inquiries, medical investigations and criminal 
for various applications such as prediction systems. Also facial expressions training 
can also be used in field applications. The use of automatic emotion recognition; digital 
advertising, marketing analysis, online games, customer feedback evaluation, commercial, 
military, security, social applications, various intelligent systems such as healthcare (e-health, 
learning, suggestion for tourism, smart city, smart conversation etc.) has great potential. Additional 
as analyzing a user's emotional reactions during interaction with products or services 
to evaluate the effectiveness of advertising campaigns and target audiences. Evaluate the effectiveness of advertising campaigns and target audiences 
can be used to develop appropriate marketing strategies. In the field of education, students 
can be used to monitor their emotional state and optimize the training process.
In machine learning, various facial expression recognition algorithms for emotion detection 
has been proposed. However, the complexity, diversity, overlap, illumination 
problems and other difficulties in facial expression recognition, recognition in practical applications 
accuracy is still not satisfactory. In recent years, neural networks 
developments have led to effective models of deep learning and a large number of deep learning 
led to the development of architecture.
One of these areas is facial expression recognition, computerized 
has an important role in vision and artificial intelligence. Deep learning, emotion recognition 
has shown real promise in terms of classification efficiency for problems. In particular 
Deep convolutional networks have revolutionized the processing of images, video and audio.
For the detection, segmentation and recognition of objects and regions in images 
is being implemented with great success.
Examples include traffic sign recognition, segmentation of biological images and natural 
tagged images, such as detection of faces, text, pedestrians and human bodies 
applications where data is abundant, autonomous robots or autonomous vehicles. CNN's 
The biggest recent success is facial recognition. From a person's gestures and facial expressions
By examining his face, it is easy to understand what kind of emotion he has. With curiosity 
data usage is activated. For the benefit of users and for use in all sectors 
is an important study that can find a field. Considering all these benefits, sentiment analysis is both 
is both today's and the future's workspaces and is becoming increasingly indispensable. 
technology is predicted.

<br>

**4.3 Potential Target Users**
Emotion analysis projects with human faces have been conducted in the fields of commerce, military, manufacturing, health and 
is interesting for service sectors. What are the companies about themselves now? 
They want to know what is being thought and act accordingly. In this way, deep 
learning-enabled systems in research areas such as economics, marketing, politics, etc. 
rich source of data that can be utilized. This situation has led to the development of 
has helped the subject to gain importance day by day. Emotion from facial expressions 
At the advanced stage of recognition, the concept of text mining comes to the fore.
The deep learning supported emotion recognition project has a very dense target user base. 
has. There are many sectors mentioned in this audience. Deep learning supported emotion recognition 
project is integrated into camera systems, providing a rich source of data. 
is being collected. Processing this data source provides a wide range of information. A school 
system, while information about education is integrated into a military system. 
makes it easier to perform crime analysis through emotion detection. A 
If the trading company integrates the system in the store where it does its marketing, customers will have access to different products. 
by analyzing the emotional attitudes it creates towards brands. Brands need to be strategic, fast 
and help them make more informed marketing and product development decisions. This way 
will be ensured to demonstrate a pragmatic attitude towards its customers and internal accounting. Project 
if connected to a hospital camera system, it will detect the early detection of certain diseases and 
facilitate the identification of newly developing findings. Utilitarianism towards the school system 
The effect can be explained as follows: the emotional states of the students according to the courses 
are recorded. In this way, the attitudes of age groups towards the lessons and the difficulty of the subjects are recorded. 
information such as levels can be learned. In addition, it is possible to learn the international 
whether or not it appeals to the palate of a community of people. 
Determination of the state of mind of the society, film evaluations, market-price balance analysis, 
Scientific and medical research, Crime analysis, security, intelligence, etc., both general and 
potential and target users of the project 
include educators, psychologists, health professionals and communication specialists. 
Machine Interaction and Sentiment Analysis are used in marketing and advertising strategies, customer 
In analyzing the experience, various sub-sectors in Safety and Monitoring, Healthcare and social sciences 
can be used and developed in the applications of the fields. Counting all possible evaluations 
Unfortunately, it is not over. These systems, which are predicted to have the potential to be used in every field 
is not just a prediction of the future because today we are not able 
It is also known to be utilized.

<br>
<br>

![Ekran Görüntüsü (102)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/f560dbaa-2adb-4e6d-8541-1549dc06d64b)
![Ekran Görüntüsü (103)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/b22230dd-e68c-4ba4-b43c-efc1a7a48660)
![Ekran Görüntüsü (104)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/d885fb4f-0ef9-4439-9b6f-a485408b4ab2)
![Ekran görüntüsü 2023-06-11 163034](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/4787e6f7-1c60-4fea-ace5-f6d5e4d04ebb)

![Ekran Görüntüsü (106)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/0e1a1911-2a6c-4e11-8672-b925a7f60399)
![Ekran görüntüsü 2023-06-11 170327](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/bcb8dac7-2ddc-4f8c-b789-d985dc287257)
![Ekran görüntüsü 2023-06-11 170351](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/8492315d-48a1-479f-977e-477ad0860de6)


## Cleanup / Deprecated Scripts (2025)
Legacy duplicate scripts (`example.py`, `xyz.py`, `main_anaconda.py`) have been deprecated and replaced with lightweight stubs to reduce confusion. Use these entry points:

- `app.py` : Streamlit interface (image upload + optional webcam, aggregates last 20 frames)
- `gradioo.py` : Gradio interface (tabs for webcam and image)
- `main.py` : Minimal OpenCV demo + playlist redirection after fixed capture window

Deprecated stubs remain only for historical reference and will exit immediately if run.



