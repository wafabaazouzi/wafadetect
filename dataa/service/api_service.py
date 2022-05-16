import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer







cv = pickle.load(open("../pickles/_vectorizer.pickle", "rb")) 
le = pickle.load(open("../pickles/_le.pickle", "rb")) 
loaded_model = pickle.load(open('../pickles/_model.pickle','rb'))


def predict(txt):
    texte = cv.transform([txt]).toarray()  # convert text to bag of words model (Vector)
    lang = loaded_model.predict(texte)  # predict the language
    lang = le.inverse_transform(lang)  # find the language corresponding with the predicted value

    return lang[0]



