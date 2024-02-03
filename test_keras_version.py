import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from timeit import timeit
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger("Keras_Performance")

def load_data():
  data, labels = fetch_20newsgroups(return_X_y=True)
  trainset_size = int((80 * len(data)) // 100) # 80% train set, 20% test set
  train_set = pd.DataFrame({"data":data[0:trainset_size], "labels":labels[0:trainset_size]})
  test_set = pd.DataFrame({"data":data[trainset_size:], "labels":labels[trainset_size:]})
  return train_set, test_set

class Model(ABC):
  import keras
  @abstractmethod
  def get_name(self):
    pass

  @abstractmethod
  def model(self, Accelerator):
    pass

  @abstractmethod
  def train(self, data: list[str]) -> float:
      pass

  @abstractmethod
  def predict(self, data: list[str]) -> float:
      pass

class Bayesian(Model):
  # non keras model, with no useage of GPU or TPU, should remain constant \ set a baseline
  def __init__(self, Accelerator):
    self.model = MultinomialNB()
    self.vectorizer = CountVectorizer()

  def get_name(self):
    return "Bayesian"

  def model(self, Accelerator):
    # this model doesn't leverage accelerators TODO: verify
    pass

  def train(self, data: list[str]) -> float:
    X_train = self.vectorizer.fit_transform(data)
    print(X_train) # TODO: remove

  def predict(self, data: list[str]) -> float:
    X_test = self.vectorizer.transform(data)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(data.labels, y_pred)
    return inference_results
    print(accuracy) # TODO: remove




# class BiLstm(Model):
#   import keras
#   from keras import layers
#   max_features = 20000  # Only consider the top 20k words
#   maxlen = 500  # Only consider the first maxlen words

#   def get_name(self):
#     return "BiLSTM"

#   def model(self, Accelerator):
#     inputs = keras.Input(shape=(None,), dtype="int32")
#     x = layers.Embedding(max_features, 128)(inputs)
#     x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
#     x = layers.Bidirectional(layers.LSTM(64))(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
#     model = keras.Model(inputs, outputs)
#     model.summary()
# # add categorical layer 3 -> num of categories
# # change keras struct to new struct
# # update to catecgorical loss function
# model.add(Dense(3))
# model.add(Activation('softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# # save model to self.model


#   def train(self, data: list[str]) -> float:
#     # inaccurate under here
#     # x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
#     # x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)
#     # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     # model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))


#   def predict(self, data: list[str]) -> float:
#     raise NotImplemented

class GPT2(Model):
  def __init__(self, Accelerator):
    pass

  def get_name(self):
    return "GPT2"

  def model(self, Accelerator):
    raise NotImplemented

  def train(self, data: list[str]) -> float:
    raise NotImplemented

  def predict(self, data: list[str]) -> float:
    raise NotImplemented

def main():
    models = [Bayesian]  # , BiLSTM, GPT2]
    column_names = ["keras_version", "model", "GPU", "train_duration", "inference_duration"]
    Acceleration = ["CPU", "GPU"]
    REPETITIONS = 2

    train_set, test_set = load_data()
    results = pd.DataFrame(columns=column_names)

    for Accelerator in Acceleration:
        for model in models:
            logger.info(f"Running {model} with {Accelerator}")
            model_instance = model(Accelerator)
            train_duration = timeit(model_instance.train(train_set), number=REPETITIONS)
            inference_duration = timeit(model_instance.predict(test_set), number=REPETITIONS)
            current_results = {"keras_version": keras_version,
                               "model": model.get_name(),
                               "Acceleration": Accelerator,
                               "train_duration": train_duration,
                               "inference_duration": inference_duration}
            results = results.append(current_results, ignore_index=True)
            logger.info(f"Running {model} finished training in {train_duration} and inference in {inference_duration}")
    print(results)


if __name__ == "__main__":
    main()