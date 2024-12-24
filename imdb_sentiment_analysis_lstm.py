import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import spacy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer

num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

maxlen = 500
x_train_pad = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test_pad = pad_sequences(x_test, padding='post', maxlen=maxlen)

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

sample_text = "I really loved the movie, it was great!"
processed_text = preprocess_text(sample_text)
print(f"Processed Text: {processed_text}")

model = models.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    layers.Bidirectional(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(x_train_pad, y_train, epochs=5, batch_size=64, validation_data=(x_test_pad, y_test), callbacks=[early_stopping])
model.save('imdb_sentiment_model.h5')
print("Model saved successfully!")

y_pred = (model.predict(x_test_pad) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
loaded_model = load_model('imdb_sentiment_model.h5')
new_review = "I absolutely loved this film, it was amazing!"
new_review_processed = preprocess_text(new_review)  


word_index = imdb.get_word_index()
new_review_seq = [word_index.get(word, 0) for word in new_review_processed.split()]
new_review_seq_padded = pad_sequences([new_review_seq], maxlen=maxlen, padding='post')

prediction = loaded_model.predict(new_review_seq_padded)
print(f"Prediction for the review: {new_review}")
print("Sentiment: Positive" if prediction > 0.5 else "Sentiment: Negative")
