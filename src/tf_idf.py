import joblib

#load the model and vectorizer for use
model=joblib.load("models/TF-IDF/model.pkl")
vectorizer=joblib.load("models/TF-IDF/vectorizer.pkl")


def predict_trait():
    text=input("Enter some text that best fits your character: ")
    if not text.strip():
        print("Input valid text")
        predict_trait()
        return
    vectorized_text=vectorizer.transform([text])
    prediction=model.predict(vectorized_text)
    print(f"The text exhibits {prediction[0].capitalize()} personality trait.")






if __name__=="__main__":
    predict_trait()