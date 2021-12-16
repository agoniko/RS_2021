import os
import re
import string
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import keras
from sanic import Sanic
from sanic.response import json as sanicjson
import mysql.connector as mysqldb
import itertools
from scipy import spatial
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, Input

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub
import tensorflow as tf
from nltk import download as ntdownload
from tqdm import tqdm
import shutil
import logging
from src.services.Google_storage_service import Gstorage

ntdownload('stopwords')
ntdownload('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def now():
    return datetime.now().strftime('%Y-%m-%d')


def safe_run(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('Something happened @', func.__name__, ':')
            print(str(e))
            return None
        return func_wrapper


class RS:
    def __init__(self):
        self.debug = False
        self.tf_hub_path = "resources/tf_hub_model/"
        self.export_root_path = 'resources/models-deployed/'
        if not os.path.exists(self.export_root_path):
            os.mkdir(self.export_root_path)

        self.db_params = dict(

            ### DEMO ELIA
            #host='52.174.111.67',
            #port=3306,
            #db='lms_2020',
            #user='whoteacher',
            #passwd='wh0t3chpass'

            ### DEMO WT
            host='40.119.137.9',
            port=3306,
            db='lms_2020',
            user='recommend',
            passwd='VmzLSSj4YkGcwxOp'
        )

        self.dbconn = None
        self.data_df = None

        self._load_mappings()

    def _dbConnect(self):
        self.dbconn = mysqldb.connect(**self.db_params)

    def _dbDisconnect(self):
        self.dbconn.close()

    def _fetchData(self, query):
        self.data_df = pd.io.sql.read_sql(query, con=self.dbconn)

    def _fetchDataCB(self, query):
        self.data_keywords = pd.io.sql.read_sql(query, con=self.dbconn)

    def _fetchDataMerlot(self, query):
        self.data_keywords_merlot = pd.io.sql.read_sql(query, con=self.dbconn)

    def _fetchDataNegative(self, query):
        self.data_to_negative = pd.io.sql.read_sql(query, con=self.dbconn)

    def _fetchDataWS(self, query):
        self.data_keywords_ws = pd.io.sql.read_sql(query, con=self.dbconn)

    # @safe_run
    def _load_mappings(self):
        ### List all exported/model files
        files = os.listdir(self.export_root_path)

        ### We want to load the latest ones. Sort ensures sorting thanks to filename
        model = sorted(filter(lambda f: 'model' in f, files))
        users = sorted(filter(lambda f: 'users' in f, files))
        resource = sorted(filter(lambda f: 'resource' in f, files))


        ### Load them
        if len(model):
            self.latest_model = load_model(self.export_root_path + model[0])

        if len(users):
            self.user_map_encode = pd.read_csv(self.export_root_path + users[0])
            self.user_map_encode = dict(self.user_map_encode.values)

        if len(resource):
            self.resource_map_encode = pd.read_csv(self.export_root_path + resource[0])
            self.resource_map_encode = dict(self.resource_map_encode.values)

            ### Map back results: very bad behavior here
            self.resource_map_encode_reversed = {value: key for (key, value) in self.resource_map_encode.items()}

    ############### Training RS Users ###############
    def _cf_training(self):
        ### Paths + date
        base_path = self.export_root_path
        model_filepath = base_path + 'model-ratings.h5'
        usersdict_filepath = base_path + 'users-dict-encoded.csv'
        resourcedict_filepath = base_path + 'resource-dict-encoded.csv'

        def map_label(x):
            if x == 1:
                return 0.1
            elif x == 2:
                return 0.3
            elif x == 3:
                return 0.5
            elif x == 4:
                return 0.7
            else:
                return 0.9

        map_label_vectorized = np.vectorize(map_label)

        ### Preparing dataset for training
        user_enc = LabelEncoder()
        self.data_df['userid_enc'] = user_enc.fit_transform(self.data_df['userid'].values)
        n_users = self.data_df['userid_enc'].nunique()

        item_enc = LabelEncoder()
        self.data_df['resource_enc'] = item_enc.fit_transform(self.data_df['resource'].values)
        n_books = self.data_df['resource_enc'].nunique()

        self.data_df['rating'] = self.data_df['rating'].values.astype(np.float32)

        ### Saving dict correspondence...needed in predict
        self.data_df[['userid', 'userid_enc']].drop_duplicates().to_csv(usersdict_filepath, index=False)
        self.data_df[['resource', 'resource_enc']].drop_duplicates().to_csv(resourcedict_filepath, index=False)

        ### X, y
        X = self.data_df[['userid_enc', 'resource_enc']].values
        y = self.data_df['rating'].values
        y = map_label_vectorized(y)

        ### NN parameters
        embedding_size = 100  # 100
        dim_first = 200  # 50
        dim_second = 100
        epochs_n = 50
        batch_n = 256 #1024
        seed = 9 #7
        np.random.seed(seed)
        kf = KFold(n_splits=10, random_state=seed, shuffle=True)

        ### NN layers
        user_input_c = Input(shape=[1], name='user')
        item_input_c = Input(shape=[1], name='item')

        embedding_u_c = keras.layers.Embedding(n_users, embedding_size, name='Embedding_user')(user_input_c)
        embedding_b_c = keras.layers.Embedding(n_books, embedding_size, name='Embedding_book')(item_input_c)

        flatten_u_c = Flatten(name='Flatten_user')(embedding_u_c)
        flatten_b_c = Flatten(name='Flatten_book')(embedding_b_c)

        # prima formula
        user_vecs_c = Dense(dim_first, activation='relu')(flatten_u_c)
        item_vecs_c = Dense(dim_first, activation='relu')(flatten_b_c)
        input_vecs_c = concatenate([user_vecs_c, item_vecs_c], name='Concat')
        e_l_u_r = Dense(dim_first * 2, activation='relu')(input_vecs_c)
        drop_1 = Dropout(0.3)(e_l_u_r)
        # seconda formula
        alfa_l_u_r = Dense(1, activation='softmax')(drop_1)

        # terza formula
        hu_l_1 = alfa_l_u_r * user_vecs_c
        hr_l_1 = alfa_l_u_r * item_vecs_c

        x_u_c = Dense(dim_first, activation='sigmoid')(hu_l_1)
        x_r_c = Dense(dim_first, activation='sigmoid')(hr_l_1)
        concat_l_1 = concatenate([x_u_c, x_r_c], name='Concat_l_1')
        x_c = Dense(dim_second, activation='relu')(concat_l_1)
        """
        x_c2 = Dense(50, activation='relu')(x_c)
        """
        y_c = full_c = Dense(1, activation='sigmoid')(alfa_l_u_r)#(x_c2)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, amsgrad=False)
        es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=5)

        model_c = Model(inputs=[user_input_c, item_input_c], outputs=y_c)
        model_c.compile(loss='mse', optimizer=adam, metrics=['mse'])

        ### NN fit
        model_c.fit([X[:, 0], X[:, 1]], y, epochs=epochs_n, batch_size=batch_n, verbose=0, validation_split=0.2,
                    shuffle=True, callbacks=[es])

        ### Locally save model
        model_c.save(model_filepath)

        gstorage = Gstorage()
        gstorage.upload_model_and_csv()

        ### Update current model
        self._load_mappings()

    # @safe_run
    def training_pipeline(self):
        self._dbConnect()
        query = f'SELECT userid,resource,rating FROM mdl_block_rate_resource'
        self._fetchData(query)
        self._dbDisconnect()
        self._cf_training()  # collaborative filtering su valutazioni


    ############### Training RS - Content Based ###############
    def _preprocessing(self):
        items = self.data_keywords
        low_case_item = items['keywords'].apply(lambda x: x.lower())
        remove_numbers = low_case_item.apply(lambda x: re.sub(r'\d+', '', x))
        replace_chars = remove_numbers.apply(lambda x: x.replace(',', ' '))
        replace_chars = replace_chars.apply(lambda x: x.replace('-', ' '))
        remove_punctuation = replace_chars.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        remove_whitespaces = remove_punctuation.apply(lambda x: x.strip())
        tokenizer = RegexpTokenizer('\w+|\$[\d]+|\w+')
        tokenization = remove_whitespaces.apply(lambda x: tokenizer.tokenize(x))
        stop_words = set(stopwords.words('english'))  # todo: add italian
        remove_stop_words = tokenization.apply(lambda x: [i for i in x if not i in stop_words])
        lemmatizer = WordNetLemmatizer()
        lemmatizer_word = remove_stop_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
        items['p_keywords'] = lemmatizer_word

    # create matrix weighted with TF IDF
    def _TF_IDF(self):
        data = self.data_keywords
        data['proc'] = data['p_keywords'].apply(lambda x: ' '.join(x))
        tf_idf = TfidfVectorizer()
        tfidf_matrix = tf_idf.fit_transform(data['proc'])
        latent_matrix = pd.DataFrame(tfidf_matrix.toarray(), index=data.index.tolist())
        return latent_matrix

    def _get_svd_embeddings(self, feature_matrix, n):
        svd = TruncatedSVD(n_components=n)
        latent_matrix = svd.fit_transform(feature_matrix)
        latent_df = pd.DataFrame(latent_matrix, index=feature_matrix.index.tolist())
        return latent_df

    # restituisce l'ID delle risorse
    def _recommender_cb(self, matrix, n):
        items = self.data_keywords

        similarity = cosine_similarity(matrix, matrix[-1:]).squeeze(1)

        similarity_sort = similarity.argsort()[::-1]
        similarity_sort = similarity_sort[1:n + 1]

        data = items[items.index.isin(similarity_sort)]['Id_resource'].values.astype(np.int32)

        similarity = -np.sort(-similarity)  # reverse sort
        similarity = similarity[1:n + 1]

        ### MinMax normalize
        similarity = similarity + np.abs(similarity.min())
        similarity = similarity / np.abs(similarity).max()

        return list(zip(map(str, data), similarity))

    def _manager_dataframe(self, user):
        self.data_keywords = self.data_keywords.query("Property == 'keywords'")[['Id_resource', 'Value']].dropna(
            subset=['Id_resource', 'Value'])
        self.data_keywords = self.data_keywords.groupby('Id_resource')['Value'].apply(lambda x: ','.join(x))
        self.data_keywords = pd.DataFrame(self.data_keywords.items(), columns=['Id_resource', 'keywords'])
        self.data_keywords = self.data_keywords.append(user, ignore_index=True)

    # @safe_run
    def content_based_pipeline(self, keywords, n=5):
        user_profile = {'Id_resource': -1, 'keywords': ','.join(keywords)}

        ### Load data
        self._dbConnect()
        query = f'SELECT Id_resource,Value,Property FROM mdl_metadata'
        self._fetchDataCB(query)
        self._dbDisconnect()

        ### Insert user profile in dataset keywords
        self._manager_dataframe(user_profile)

        ### Preprocessing keywords
        self._preprocessing()
        print('preprocessig fatto')
        ### create weighted matrix with tf idf
        latent_matrix = self._TF_IDF()

        ### Apply SVD over matrix TF-IDF
        # 25 is the size of the vector representation
        matrix_svd = self._get_svd_embeddings(latent_matrix, 100)

        ### call function reccomendation
        # 5 is number of items to return
        return self._recommender_cb(matrix_svd, n)

    ############### Predict RS - Users ###############
    def _collaborative_filtering_recommend(self, userid, n):
        ### Load mappings
        user_map_encode = 0

        ### Get resource _not_ yet evaluated by the user (all_ids - evaluated_ids)
        resources_to_exclude = self.data_df[self.data_df.userid == userid]
        resources_to_exclude = resources_to_exclude.resource
        resource_to_evaluate = set(self.data_df.resource) - set(resources_to_exclude)

        ### Map resources with mapping
        X_user_encoded = self.user_map_encode[int(userid)]
        X_resources_encoded = list(map(lambda r: self.resource_map_encode[r], resource_to_evaluate))

        ### Predictable data structure
        X = [[X_user_encoded, resource] for resource in X_resources_encoded]
        X = np.array(X)

        ### Predict
        prediction = self.latest_model.predict([X[:, 0], X[:, 1]])

        ### Assign prediction to resources
        prediction = list(zip(X[:, 1], prediction[:, 0]))

        ### Sort by rating predicted
        prediction = sorted(prediction, key=lambda k: k[1], reverse=True)

        ### Keet best 5
        prediction_top = prediction[:n]

        ### salvo raccomandazioni non realizzate
        self.data_to_negative_rec = [[str(self.resource_map_encode_reversed[int(r[0])]), float(r[1])] for r in prediction[n:]]

        ### Just to get back real values
        result = [[str(self.resource_map_encode_reversed[int(r[0])]), float(r[1])] for r in prediction_top]

        return result

    # @safe_run
    def collaborative_filtering_pipeline(self, userid, n):
        userid = int(userid)

        ### Load data
        self._dbConnect()
        query = f'SELECT userid,resource FROM mdl_block_rate_resource'
        self._fetchData(query)
        self._dbDisconnect()

        ### Get recommendation
        return self._collaborative_filtering_recommend(userid, n)

    def negative_recommend(self, old_rec, users, n):
        self._dbConnect()
        query = f'SELECT userid, resource, rating FROM mdl_block_rate_resource'
        self._fetchDataNegative(query)
        self._dbConnect()
        data = self.data_to_negative
        #elimino risorse già raccomandate
        rac_other_sections = list(set(list(old_rec.keys())))
        ###
        data = data[data['userid'].isin(list(map(int, users)))]
        risorse_valutate_utenti = set(data.resource.values)
        possibili_risorse_to_rec = set([int(item[0]) for item in self.data_to_negative_rec])
        print(set([int(item) for item in rac_other_sections]))
        possibili_risorse_to_rec = possibili_risorse_to_rec.difference(set([int(item) for item in rac_other_sections]))
        risorse_comuni = risorse_valutate_utenti & possibili_risorse_to_rec
        data_to_mean = data[data['resource'].isin(list(risorse_comuni))]
        data_to_mean = data_to_mean[['resource','rating']]
        data_to_mean.rating = data_to_mean.rating/5
        mean_realized = data_to_mean.groupby('resource')['rating'].mean()
        rating_predicted = {}
        for tupla in self.data_to_negative_rec:
            rating_predicted[tupla[0]] = tupla[1]
        dict_means = mean_realized.to_dict()
        dict_rate_final = {}
        for key in dict_means.keys():
            dict_rate_final[key] = dict_means[key] - rating_predicted[str(key)]
        final_result = {}
        for key in dict_rate_final.keys():
            if dict_rate_final[key] > 0:
                final_result[str(key)] = dict_rate_final[key]
        #print("neg--------------------------", final_result)
        #sort_dict = dict(sorted(final_result.items(), key=lambda item: item[1]))
        return dict(itertools.islice(final_result.items(), n))

    def elementi_nuovi_rec(self,cb_cf,rn,number):
        self._dbConnect()
        query = f'SELECT userid, resource, rating FROM mdl_block_rate_resource'
        self._fetchDataNegative(query)
        self._dbConnect()
        data = self.data_to_negative
        data_high_rating = data.query("rating > 3")

        rac_other_sections = list(set(list(cb_cf.keys()) + list(rn.keys())))
        data_to_rec = data_high_rating[~data_high_rating['resource'].isin(rac_other_sections)]
        data_count = data_to_rec.groupby('resource')['rating'].count()
        data_count = data_count.to_dict()
        final_element = {}
        for element in data_count.keys():
            if data_count[element] < 3:
                final_element[str(element)] = data_count[element]
        sort_dict = dict(sorted(final_element.items(), key=lambda item: item[1]))
        return dict(itertools.islice(sort_dict.items(), number))

    def content_based_weschool_db(self, keywords, n=5):
        user_profile = {'id': 999999, 'link': 'noLink', 'keywords': ','.join(keywords)}
        ### Load data
        self._dbConnect()
        query = f'SELECT * FROM mdl_weschool_data'
        self._fetchDataWS(query)
        self._dbDisconnect()
        ### Insert user profile in dataset keywords
        self.data_keywords_ws = self.data_keywords_ws[['id', 'keywords']]
        self.data_keywords_ws = self.data_keywords_ws.append(user_profile, ignore_index=True)
        ### Preprocessing keywords
        items = self.data_keywords_ws
        low_case_item = items['keywords'].apply(lambda x: x.lower())
        remove_numbers = low_case_item.apply(lambda x: re.sub(r'\d+', '', x))
        replace_chars = remove_numbers.apply(lambda x: x.replace(',', ' '))
        replace_chars = replace_chars.apply(lambda x: x.replace('-', ' '))
        remove_punctuation = replace_chars.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        remove_whitespaces = remove_punctuation.apply(lambda x: x.strip())
        tokenizer = RegexpTokenizer('\w+|\$[\d]+|\w+')
        tokenization = remove_whitespaces.apply(lambda x: tokenizer.tokenize(x))
        stop_words = set(stopwords.words('english'))  # todo: add italian
        remove_stop_words = tokenization.apply(lambda x: [i for i in x if not i in stop_words])
        lemmatizer = WordNetLemmatizer()
        lemmatizer_word = remove_stop_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
        items['p_keywords'] = lemmatizer_word

        ### create weighted matrix with tf idf
        data = self.data_keywords_ws
        data['proc'] = data['p_keywords'].apply(lambda x: ' '.join(x))
        tf_idf = TfidfVectorizer()
        tfidf_matrix = tf_idf.fit_transform(data['proc'])
        latent_matrix = pd.DataFrame(tfidf_matrix.toarray(), index=data.index.tolist())
        ### Apply SVD over matrix TF-IDF
        # 25 is the size of the vector representation
        matrix_svd = self._get_svd_embeddings(latent_matrix, 250)

        ### call function reccomendation
        # 5 is number of items to return
        matrix = matrix_svd
        # items = self.data_keywords_merlot

        similarity = cosine_similarity(matrix, matrix[-1:]).squeeze(1)

        similarity_sort = similarity.argsort()[::-1]
        similarity_sort = similarity_sort[1:n + 1]

        data = items[items.index.isin(similarity_sort)]['id'].values.astype(np.int32)

        similarity = -np.sort(-similarity)  # reverse sort
        similarity = similarity[1:n + 1]

        ### MinMax normalize
        similarity = similarity + np.abs(similarity.min())
        similarity = similarity / np.abs(similarity).max()

        return list(zip(map(str, data), similarity))


    # @safe_run
    def content_based_merlot_db(self, keywords, n=5):
        user_profile = {'id': 999999, 'link': 'noLink', 'keywords': ','.join(keywords)}

        ### Load data
        self._dbConnect()
        query = f'SELECT * FROM mdl_merlot_data'
        self._fetchDataMerlot(query)
        self._dbDisconnect()
        ### Insert user profile in dataset keywords
        self.data_keywords_merlot = self.data_keywords_merlot[['id', 'link', 'keywords']]
        self.data_keywords_merlot = self.data_keywords_merlot.append(user_profile, ignore_index=True)

        ### Preprocessing keywords
        items = self.data_keywords_merlot
        low_case_item = items['keywords'].apply(lambda x: x.lower())
        remove_numbers = low_case_item.apply(lambda x: re.sub(r'\d+', '', x))
        replace_chars = remove_numbers.apply(lambda x: x.replace(',', ' '))
        replace_chars = replace_chars.apply(lambda x: x.replace('-', ' '))
        remove_punctuation = replace_chars.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        remove_whitespaces = remove_punctuation.apply(lambda x: x.strip())
        tokenizer = RegexpTokenizer('\w+|\$[\d]+|\w+')
        tokenization = remove_whitespaces.apply(lambda x: tokenizer.tokenize(x))
        stop_words = set(stopwords.words('english'))  # todo: add italian
        remove_stop_words = tokenization.apply(lambda x: [i for i in x if not i in stop_words])
        lemmatizer = WordNetLemmatizer()
        lemmatizer_word = remove_stop_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
        items['p_keywords'] = lemmatizer_word

        ### create weighted matrix with tf idf
        data = self.data_keywords_merlot
        data['proc'] = data['p_keywords'].apply(lambda x: ' '.join(x))
        tf_idf = TfidfVectorizer()
        tfidf_matrix = tf_idf.fit_transform(data['proc'])
        latent_matrix = pd.DataFrame(tfidf_matrix.toarray(), index=data.index.tolist())

        ### Apply SVD over matrix TF-IDF
        # 25 is the size of the vector representation
        matrix_svd = self._get_svd_embeddings(latent_matrix, 100)

        ### call function reccomendation
        # 5 is number of items to return
        matrix = matrix_svd
        # items = self.data_keywords_merlot

        similarity = cosine_similarity(matrix, matrix[-1:]).squeeze(1)

        similarity_sort = similarity.argsort()[::-1]
        similarity_sort = similarity_sort[1:n + 1]

        data = items[items.index.isin(similarity_sort)]['id'].values.astype(np.int32)

        similarity = -np.sort(-similarity)  # reverse sort
        similarity = similarity[1:n + 1]

        ### MinMax normalize
        similarity = similarity + np.abs(similarity.min())
        similarity = similarity / np.abs(similarity).max()

        return list(zip(map(str, data), similarity))

    def load_or_encode_merlot_metadata(self,txts):
        ###load or encode encoded_articles
        gstorage = Gstorage()
        try:
            gstorage.download_file("encoded_articles/encoded_articles.txt", "resources/encoded_articles/encoded_articles.txt")
            encoded_txts = np.loadtxt("resources/encoded_articles/encoded_articles.txt", delimiter=',')
            if (len(encoded_txts) != len(txts)):
                raise FileNotFoundError
            else:
                return encoded_txts
        except (FileNotFoundError, IOError):
            print("Encoded resources not found or not up to date, encoding now...")
            try:
                os.makedirs('resources/encoded_articles')
            except OSError:
                print("resources directory already exists")
            embed = self.load_tf_hub_model()
            encoded_articles = [embed(txt).numpy()[0] for txt in tqdm(txts)]
            np.savetxt("resources/encoded_articles/encoded_articles.txt", encoded_articles, delimiter=',')
            gstorage.upload_file("encoded_articles/encoded_articles.txt", "resources/encoded_articles/encoded_articles.txt")
            return encoded_articles



    def get_cosine_similarity(self,encoded_text1, encoded_text2):
        return 1 - spatial.distance.cosine(encoded_text1, encoded_text2)


    def load_tf_hub_model(self):
        embed = None
        while embed == None:
            try:
                logging.warning("trying to load the model")
                embed = hub.load(self.tf_hub_path)
                return embed
            except:
                logging.warning("removing damaged directory")
                try:
                    shutil.rmtree(self.tf_hub_path)
                except OSError:
                    logging.warning("directory not found")

                logging.debug("downloading the model")
                embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
                tf.saved_model.save(embed, self.tf_hub_path)
                return embed




    # @safe_run
    def content_based_merlot_db_cosine(self, text ,n=5):

        if type(text) == list:
            text = ' '.join(text)

        ### Load data
        self._dbConnect()
        query = f'SELECT * FROM mdl_merlot_data'
        self._fetchDataMerlot(query)
        self._dbDisconnect()

        ###append feature to obtain a unique text
        df = self.data_keywords_merlot
        df['disciplines'] = df['disciplines'].apply(lambda x: x.replace('/', ' '))
        df.reset_index(inplace=True,drop=True)
        txts = df['disciplines'] + ' ' + df['title'] + ' ' + df['keywords'] + ' ' + df['description']
        txts = txts.apply(lambda x: x.lower())
        txts = txts.apply(lambda x: " ".join(list(set(x.split()))))


        ###If the encoding file is up to date the encodings will be loaded in the var, otherwise they will be encoded from the beginning
        encoded_txts = self.load_or_encode_merlot_metadata(txts)

        ###embedding input text
        embed = self.load_tf_hub_model()
        encoded_text = embed(text)

        ###get cosine similarity
        df['cosine'] = [self.get_cosine_similarity(encoded_text, encoded_txts[i]) for i in range(len(encoded_txts))]
        scores = df.sort_values(by='cosine', ascending=False).head(n)
        ###rounding scores and returning results
        scores['cosine'] = scores['cosine'].apply(lambda x: round(x, 3))

        return dict(zip(scores['id'], scores['cosine']))




### Todo:
# at loading time, load model (beware of trainings/day changes)
# add api key for security rezzonz
# exaustive loggings
# comments ;D

def makeSanicResponse(resp):
    print(now(), resp)
    return sanicjson(resp, headers={'Access-Control-Allow-Origin': '*'})


if __name__ == '__main__':
    rs = RS()
    app = Sanic('STRS')
    gstorage = Gstorage()
    gstorage.download_files()

    @app.route('/api/rs', methods=['POST','OPTIONS'])
    async def main(request):

        req_parameters = request.json
        print('////////////////////////- request received:////////////////////////////////////', req_parameters)

        try:

            #####Recommend Cosine#####
            if req_parameters['type'] == 'cosine':

                try:
                    n = int(req_parameters['n'])
                except KeyError:
                    n = 5
                    print(f"n not specified, defaulting to {n}")
                return makeSanicResponse(rs.content_based_merlot_db_cosine(req_parameters['keywords'],n))


            ########## Ping ##########
            if req_parameters['type'] == 'ping':
                return makeSanicResponse({'response': 'I recommend to use a recommender'})

            ########## Training ##########
            if req_parameters['type'] == 'training':
                rs.training_pipeline()
                return makeSanicResponse({'response': 'Training completed'})

            ########## Call RS ##########
            if req_parameters['type'] == 'recommend':
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! richiesta raccomandazione avviata !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                ### merlot
                print("********* avvio raccomandazione *********");
                cb_merlot = rs.content_based_merlot_db(req_parameters['keywords'], 2)
                cb_ws = rs.content_based_weschool_db(req_parameters['keywords'], 2)
                print("********** we school **********")
                #print(cb_ws)

                ### CF
                recommendation_cf = rs.collaborative_filtering_pipeline(req_parameters['userid'], 2)

                ### CB
                cb_len = 2 if not recommendation_cf else 4  # recommendation_cf can be 0

                recommendation_cb = rs.content_based_pipeline(req_parameters['keywords'], cb_len)

                final_recommendation = OrderedDict(recommendation_cf + recommendation_cb)

                ## negative_recommend
                cf_negative_recommend = rs.negative_recommend(dict(final_recommendation), req_parameters['usersid'], 15)

                ###raccomandazione oggetti nuovi
                recommend_new_items = rs.elementi_nuovi_rec(dict(final_recommendation), dict(cf_negative_recommend), 15)
                return makeSanicResponse({'response': dict(final_recommendation), 'merlot': dict(cb_merlot), 'negative':dict(cf_negative_recommend), 'newItems':dict(recommend_new_items), 'weschool': dict(cb_ws)})


        except Exception as e:
            makeSanicResponse({'response': 'Invalid request', 'error': str(e)})


    if __name__ == '__main__':
        ### Run flask container
        app.run(
            host='0.0.0.0',
            debug=False,
            port=5001,
            # workers=4,
            # threaded=True,
            # processes=5,
            #ssl={'cert': 'certificates/cert.pem', 'key': 'certificates/privkey.pem'}
        )




