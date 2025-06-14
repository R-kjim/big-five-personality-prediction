import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import os
import numpy as np


tokenizer=None
model=None
bert_model=None

def load_model():

    global tokenizer, model, bert_model

    try:
        model_dir=os.path.join(os.getcwd(),'models/BERT/')
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # Load base BERT
        model = tf.keras.models.load_model(model_dir)

        bert_model= TFBertModel.from_pretrained('bert-base-uncased')
        # bert_model=1
        
        return
    except Exception as e:
        print(e)

def get_bert_embeddings(texts, batch_size=8):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Tokenize using tokenizer
        inputs = tokenizer(batch, return_tensors='tf', padding=True, truncation=True, max_length=128)
        # Get BERT outputs
        outputs = bert_model(inputs)
        # Average pooling across the sequence length (axis=1)
        pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    
        return pooled_output


def predict_trait():
    if not tokenizer or not model or not bert_model:
        load_model()
        predict_trait()
    text=input("Enter some text  that best describes you: ")
    if not text.strip():
        print("Text cannot be empty")
        predict_trait()

    embeddings = get_bert_embeddings([text])

    # Predict
    predictions = model.predict(embeddings)

    # Convert prediction to labels
    predicted_class = np.argmax(predictions)
    traits=['agreeableness' ,'conscientiousness','extraversion' ,'neuroticism','openness']
    print(f"The text exhibits {traits[predicted_class].capitalize()} personality trait at {predictions[0][predicted_class]:.2%} confidence level")



if __name__=="__main__":
    predict_trait()