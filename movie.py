# importing library
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import nltk
import gensim
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, Reshape,Dot, Flatten, Dense, Concatenate, LSTM,Dropout
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import Callback
# Extract Data from the Movie_dataset and Rating_dataset
def get_data():
    movie_data = pd.read_csv('/kaggle/input/dataset/alpha copy.csv',usecols=['movieId','title','genres_y','plot','tmdbId'])
    ratings_data = pd.read_csv('/kaggle/input/new-rating/new_rating.csv', usecols=['movieId','userId','rating'])
    movie_data = movie_data.set_index('movieId')
    ratings_data = ratings_data.set_index('movieId')
    combined_data = movie_data.join(ratings_data, on='movieId', how='inner')
    #combined_data.to_feather('combined_data.feather')
    train_data, test_data = train_test_split(combined_data, test_size=0.2)
    return train_data, test_data
w2v_model = gensim.models.Word2Vec.load("/kaggle/input/word2v/word2vec.model")
def create_model(n_users, n_movies, n_genres,latent_dim):
    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users+1, latent_dim, name='user_embedding')(user_input)
    user_vec = Flatten()(user_embedding)

    movie_input = Input(shape=[1])
    movie_embedding = Embedding(n_movies+1, latent_dim, name='movie_embedding')(movie_input)
    movie_vec = Flatten()(movie_embedding)

    genre_input = Input(shape=[n_genres],dtype='int16')
    genre_embedding = Embedding(n_genres, latent_dim, name='genre_embedding')(genre_input)
    genre_vec = Flatten()(genre_embedding)
    
    # Add word2vec embedding layer
    vocab_size = w2v_model.wv.vectors.shape[0]
    embedding_dim = w2v_model.wv.vectors.shape[1]
    plot_input = Input(shape=[1])
    plot_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[w2v_model.wv.vectors], trainable=False)(plot_input)
    plot_vec = Flatten()(plot_embedding)

    
    concat = Concatenate()([user_vec, movie_vec, genre_vec, plot_vec])
    concat_reshaped = Reshape((-1, latent_dim))(concat)
    
    # Using a RNN model instead of a CNN
    lstm = LSTM(32, return_sequences=True)(concat_reshaped)
    dropout = Dropout(0.5)(lstm)
    dense = Dense(32, activation='relu')(dropout)
    output = Dense(1)(dense)

    model = Model([user_input, movie_input, genre_input, plot_input], output)
    optimizer = Adamax(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
  
  def one_hot_encode_genres(data):
    genres = list(set(genre for genres in data['genres_y'] for genre in genres.split("|")))
    genres_encoder = LabelEncoder().fit(genres)
    genres_encoded = genres_encoder.transform(data['genres_y'].str.split("|").sum())
    genres_encoded = genres_encoded.reshape(-1, 1)
    genres_one_hot = OneHotEncoder().fit_transform(genres_encoded)
    data['genres_encoded'] = genres_one_hot
    return data, genres_encoder
  class AccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs={}):
        accuracy = self.model.evaluate([self.test_data['userId'], self.test_data['tmdbId'], tfidf_matrix_test, self.test_data['genres']], self.test_data['rating'])
        print("Accuracy: {:.4f}".format(accuracy))
   def vectorize_plot(plot, w2v_model):
    # Tokenize the plot
    plot_tokens = nltk.word_tokenize(plot)
    # Initialize a variable to store the plot vector
    plot_vector = np.zeros(w2v_model.vector_size)
    # Iterate over the tokens in the plot
    for token in plot_tokens:
        # Check if the token is in the word2vec model's vocabulary
        if token in w2v_model.wv:
            # Add the token's vector to the plot vector
            plot_vector += w2v_model.wv[token]
    # Divide the plot vector by the number of tokens to get the average vector
    plot_vector /= len(plot_tokens)
    return plot_vector
  def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
  def generator():
    for plot_vector in train_data['plot_vectors'].values:
        yield plot_vector
train_data, test_data=get_data()
genres_encoder = LabelEncoder()
genres = pd.concat([train_data['genres_y'], test_data['genres_y']])
genres_encoder.fit(genres)
train_data['genres_encoded'] = genres_encoder.transform(train_data['genres_y'])
############################  NLTK Vectorizer ################################################
train_data['plot_vectors'] = train_data['plot'].apply(lambda x: vectorize_plot(x, w2v_model))
train_data['plot_vectors'] = train_data['plot_vectors'].apply(lambda x: x.astype(np.float32))

print(type(train_data['plot_vectors']))

plot_vectors_ds = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
train_data['rating'] = train_data['rating'].values
n_genres = len(genres_encoder.classes_)
n_users = train_data['userId'].nunique()
n_movies = train_data['tmdbId'].nunique()
def main():
    train_data, test_data=get_data()
    train_data = reduce_mem_usage(train_data)
    test_data = reduce_mem_usage(test_data)
    ###################### One-Hot Encoder ###################################
    genres_encoder = LabelEncoder()
    genres = pd.concat([train_data['genres_y'], test_data['genres_y']])
    genres_encoder.fit(genres)
    train_data['genres_encoded'] = genres_encoder.transform(train_data['genres_y'])
    test_data['genres_encoded'] = genres_encoder.transform(test_data['genres_y'])
    ############################  NLTK Vectorizer ################################################
    
    train_data['plot_vectors'] = train_data['plot'].apply(lambda x: vectorize_plot(x, w2v_model))
    test_data['plot_vectors'] = test_data['plot'].apply(lambda x: vectorize_plot(x, w2v_model))
    train_data['plot_vectors'] = train_data['plot_vectors'].apply(lambda x: x.astype(np.float32))
    test_data['plot_vectors'] = test_data['plot_vectors'].apply(lambda x: x.astype(np.float32))
    ##############################################################################################
    plot_vectors_ds = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
    plot_vectors_ds = plot_vectors_ds.batch(1)
    plot_vectors_ds = plot_vectors_ds.map(lambda x: tf.numpy())
    plot_vectors_ds = plot_vectors_ds.unbatch()
    plot_vectors_ds = plot_vectors_ds.as_numpy_iterator()
    plot_vectors = [x for x in plot_vectors_ds]
    train_data['rating'] = train_data['rating'].values
    n_genres = len(genres_encoder.classes_)
    n_users = train_data['userId'].nunique()
    n_movies = train_data['tmdbId'].nunique()
    batch_size=16
    model = create_model(n_users, n_movies, n_genres,latent_dim=50)
    accuracy_callback = AccuracyCallback(test_data)
    #Early Stoppage Implementaion 
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    # Fitting the Model
    model.fit([train_data['userId'], train_data['tmdbId'],train_data['genres_encoded'],plot_vectors], train_data['rating'],batch_size=batch_size,epochs=30, callbacks=[accuracy_callback])
    #test_predictions = model.predict([test_data['userId'], test_data['tmdbId'], tfidf_matrix_test.toarray(), test_data['genres']])    
    #test_predictions = model.predict([test_data['userId'], test_data['tmdbId']])
    #test_predictions = np.round(test_predictions)
    #test_data['predictions'] = test_predictions
    #print(test_data.head())
    model.save('Collaborative_Model_CNN_V1.h5')
    

if __name__ == "__main__":
    
    main()
