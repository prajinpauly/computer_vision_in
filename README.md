# Music Genre Classification Using AI

## Overview

This project focuses on building a deep learning-based system for **music genre classification (MGC)**. The aim is to identify the genre of music from 30-second audio clips using extracted time-frequency features such as **Mel-spectrograms**. The solution is built using **PyTorch**, and it compares several architectures, including CNNs, CNN-LSTM, and CNN-GRU, trained on the widely recognized **GTZAN dataset**.

The goal is to explore temporal and spatial audio features and evaluate how different neural models perform in classifying music genres for real-world applications like streaming platforms, playlist-generation systems, and music therapy programs.

---

## Objectives

- Develop and evaluate deep learning models for music genre classification.
- Extract meaningful audio features using Mel-spectrograms.
- Compare multiple architectures (CNN, CNN-LSTM, CNN-GRU) to understand their strengths.
- Address challenges in training with limited datasets and improve model generalization.

---

## Applications and Relevance

Music genre classification has many important real-world applications, including:

- **Streaming Services**: Delivering accurate, personalized music recommendations.
- **Playlist Automation**: Grouping tracks by genre to enhance listening experiences.
- **Music Retrieval Systems**: Organizing large music libraries based on genre.
- **Copyright Infringement Detection**: Verifying content ownership via genre similarity.
- **Music Therapy**: Tailoring playlists for therapeutic use based on genre mood.

---

## Dataset: GTZAN Music Genre Dataset

- **Total Samples**: 1,000 audio clips  
- **Clip Length**: 30 seconds  
- **Genres**: 10 (e.g., Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock)  
- **Format**: WAV  
- **Sample Rate**: Resampled to 22,050 Hz  

Each genre has 100 clips, making the dataset balanced and suitable for benchmarking music classification models.

---

## Preprocessing & Feature Extraction

### Main Steps:
- **Mono Conversion**: All tracks converted to mono to reduce complexity.
- **Resampling**: Standardized sample rate (22,050 Hz).
- **Mel-Spectrogram Generation**:
  - `n_fft = 1024`, `hop_length = 256`, `n_mels = 128`
  - Converted to decibel scale after spectrum calculation.

### Additional Processing:
- **Uniform Chunking**: All audio clips were split into 10 equal segments of 66,000 samples.
- **Data Augmentation**: Applied on spectrograms to improve model generalization.
- **Train-Test Split**: 80% training / 20% testing data.

---

## Model Architectures

### 1. Base CNN Model

- Convolutional layers to extract spatial patterns in Mel-spectrograms.
- Pooling and dropout layers for regularization.
- Fully connected layers to perform final classification.

### 2. CNN + LSTM

- CNN layers for spatial feature extraction.
- LSTM layers to learn temporal patterns and time-based relationships in music.
- Fully connected layers to map sequence-aware features to output.

### 3. CNN + GRU

- Simplified RNN architecture using GRUs for better training speed.
- Achieved the best performance while balancing accuracy and training time.

---

## Implementation: PyTorch

- **Spectrogram Conversion**: Using Librosa and NumPy.
- **Model Definition**:
  - `torch.nn.Conv2d` for CNNs
  - `torch.nn.LSTM` and `torch.nn.GRU` for sequence learning
  - `torch.nn.Linear` for classification
- **Loss Function**: `CrossEntropyLoss` (multi-class classification)
- **Optimizer**: `Adam`, with learning rate set at 0.001
- **Training Duration**: 30 epochs with early stopping

---

## Training and Performance

| Model Variant          | Test Accuracy | Notes |
|------------------------|---------------|-------|
| Base CNN               | 79.88%        | Overfit in later epochs |
| CNN + LSTM             | 87.63%        | Better feature learning, more stable |
| CNN + GRU              | **91.29%**     | Best overall accuracy and convergence |
| Transfer Learning + Ensemble | *Applied* | Helped further stabilize predictions |

Each model was trained using GPU for accelerated learning time. Training involved data standardization, augmentation, and chunked inputs to ensure consistency.

---

## Key Learnings and Observations

- **Temporal learning** is essential for modeling music data. LSTM and GRU significantly improved classification performance over CNN alone.
- **GRU model** outperformed others with the best accuracy and convergence speed.
- **Avoiding validation split** enhanced learning efficiency given the dataset size.
- **Combining models via ensembling** further improved stability and mitigated bias.

---

## Future Improvements

To improve the system further, future versions can explore:

- **Attention-based models** or **transformers** for deeper temporal understanding.
- **Advanced feature engineering**: Including chroma features, tempo, and spectral contrast.
- **Synthetic data generation techniques** to balance training data and simulate more genre variation.
- **Regularization and stabilization** methods to reduce overfitting even further.

---

## Conclusion

This project successfully implemented and compared multiple deep learning models for music genre classification using the GTZAN dataset. Results demonstrated:

- CNN-GRU achieved the highest test accuracy (91.29%) and consistency.
- LSTM-based models effectively captured sequential features.
- Data preparation and customized architectures had a direct impact on training quality and generalization.

The insights from this project can be applied to other audio classification systems across industries such as streaming services, audio search engines, and music therapy solutions.

---

## Author

**Prajin Paul**  
📧 Email: [paulprajin2015@gmail.com](mailto:paulprajin2015@gmail.com)  
🔗 LinkedIn: [https://www.linkedin.com/in/prajin-paul-b64415247](https://www.linkedin.com/in/prajin-paul-b64415247)
