# Twitter Sentiment Analysis Web Application

A modern, responsive web application for predicting the sentiment of tweets using a trained Deep Learning model (LSTM).

## 🚀 Setup Instructions

1. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Save Your Model Files (CRITICAL STEP)**
   Your Jupyter Notebook trained the model but did not save the files. Open your notebook, train the model as usual, and **add this code block at the very end** to save your files:

   ```python
   import pickle

   # 1. Save the Keras Model
   lstm_model.save("lstm_model.h5")
   print("Model saved to lstm_model.h5")

   # 2. Save the Tokenizer
   with open("tokenizer.pkl", "wb") as f:
       pickle.dump(tokenizer, f)
   print("Tokenizer saved to tokenizer.pkl")

   # 3. Save the Label Encoder
   with open("label_encoder.pkl", "wb") as f:
       pickle.dump(le, f)
   print("Label Encoder saved to label_encoder.pkl")
   ```

3. **Organize Files**
   Once you download `lstm_model.h5`, `tokenizer.pkl`, and `label_encoder.pkl` from Colab/Jupyter, place them inside the `model/` directory of this project.

   The structure should look exactly like this:
   ```
   twitter_2.0/
   ├── app.py
   ├── preprocessing.py
   ├── requirements.txt
   ├── README.md
   ├── data/
   │   └── twitter_training.csv
   └── model/
       ├── lstm_model.h5
       ├── tokenizer.pkl
       └── label_encoder.pkl
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 🛠️ Tech Stack
- **Frontend/Backend**: Streamlit (Python)
- **Machine Learning**: TensorFlow/Keras (LSTM)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly

## 🧠 Architecture
The sentiment analysis is performed by a Bidirectional LSTM network with an Embedding layer of dimension 128, processing text padded to 40 tokens. The text preprocessing steps inside `preprocessing.py` are strictly matched to the training pipeline for perfect prediction accuracy.
