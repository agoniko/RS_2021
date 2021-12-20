import os
import re
import json
import string
import logging
import mimetypes
import numpy as np
import pandas as pd
from base64 import b64decode
from datetime import datetime
from collections import OrderedDict

from sanic import Sanic
from sanic.response import json as sanicjson
import mysql.connector as mysqldb

from ast import literal_eval
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, multiply, Input

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import download as ntdownload
ntdownload('stopwords')
ntdownload('wordnet')

from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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

		self.export_root_path = '../resources/models-deployed/'
		if not os.path.exists(self.export_root_path):
			os.mkdir(self.export_root_path)

		self.db_params = dict(
			### DEMO DB (maybe)
			#host = '13.94.150.149',
			#port = 3306,
			#db = 'sql973959_3',
			#user = 'Ricardo',
			#passwd = b64decode(b'UGlwcG8xMjM/').decode('utf-8')

			### PROD DEMO
			#host = '40.91.235.252',
			#port = 3306,
			#db = 'sql973959_3',
			#user = 'Ricardo',
			#passwd = b64decode(b'UGlwcG8xMjM/').decode('utf-8')

			### LMS RS 2020
			host = '40.119.137.9', #'52.178.97.142',
			port = 3306,
			db = 'lms_2020', #'wt_lms_rs_db2',
			user = 'recommend', #'lms_rs',
			passwd = 'VmzLSSj4YkGcwxOp' #'YUTfDKO5YyVWJ6xw'#b64decode(b'UGlwcG8xMjM/').decode('utf-8')
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

	@safe_run
	def _load_mappings(self):
		### List all exported/model files
		files = os.listdir(self.export_root_path)

		### We want to load the latest ones. Sort ensures sorting thanks to filename
		model = sorted(filter(lambda f: 'model' in f, files))
		users = sorted(filter(lambda f: 'users' in f, files))
		resource = sorted(filter(lambda f: 'resource' in f, files))

		### Load them
		if len(model):
			self.latest_model = load_model(self.export_root_path + model[-1])
			self.user_rappresentation = Model(self.latest_model.input, self.latest_model.get_layer('Embedding_user').output)
			self.item_rappresentation = Model(self.latest_model.input, self.latest_model.get_layer('Embedding_book').output)

		if len(users):
			self.user_map_encode = pd.read_csv(self.export_root_path + users[-1])
			self.user_map_encode = dict(self.user_map_encode.values)

		if len(resource):
			self.resource_map_encode = pd.read_csv(self.export_root_path + resource[-1])
			self.resource_map_encode = dict(self.resource_map_encode.values)

			### Map back results: very bad behavior here
			self.resource_map_encode_reversed = {value: key for (key, value) in self.resource_map_encode.items()}

	############### Training RS Users ###############
	def _cf_training(self):
		### Paths + date
		base_path = self.export_root_path + now()
		model_filepath = base_path + '-model-ratings.h5'
		usersdict_filepath = base_path + '-users-dict-encoded.csv'
		resourcedict_filepath = base_path + '-resource-dict-encoded.csv'
		#model_filepath_user = base_path + '-user-rappresentation.h5'
		#model_filepath_item = base_path + '-item-rappresentation.h5'

		def map_label(x):
			if   x == 1: return 0.1
			elif x == 2: return 0.3
			elif x == 3: return 0.5
			elif x == 4: return 0.7
			else:        return 0.9
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
		embedding_size = 250
		dim_first = 50
		dim_second = 25
		epochs_n = 50
		batch_n = 1024
		seed = 7
		np.random.seed(seed)
		kf = KFold(n_splits=10, random_state=seed, shuffle=True)

		### NN layers
		user_input_c  = Input(shape=[1], name='user')
		embedding_u_c = Embedding(n_users, embedding_size, name='Embedding_user')(user_input_c)
		flatten_u_c   = Flatten(name='Flatten_user')(embedding_u_c)
		user_vecs_c   = Dense(dim_first, activation='relu')(flatten_u_c) 

		item_input_c  = Input(shape=[1], name='item')
		embedding_b_c = Embedding(n_books, embedding_size, name='Embedding_book')(item_input_c)
		flatten_b_c   = Flatten(name='Flatten_book')(embedding_b_c)
		item_vecs_c   = Dense(dim_first, activation='relu')(flatten_b_c)

		input_vecs_c  = concatenate([user_vecs_c, item_vecs_c], name='Concat')

		x_c = Dense(dim_first, activation  ='relu')(input_vecs_c)
		x_c = Dense(dim_second, activation ='relu')(x_c)
		y_c = full_c = Dense(1, activation = 'sigmoid')(x_c)

		adam = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.99, amsgrad = False)
		es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=5)

		model_c = Model(inputs = [user_input_c, item_input_c], outputs = y_c)
		model_c.compile(loss = 'mse', optimizer = adam, metrics = ['mse'])

		### NN fit
		model_c.fit([X[:,0], X[:,1]], y, epochs=epochs_n, batch_size=batch_n, verbose=0, validation_split=0.1, shuffle=True, callbacks=[es])

		### Locally save model
		model_c.save(model_filepath)
		

		### Update current model
		self._load_mappings()

	
	def Similarity_NN(self, X):
		profilo_risorsa = self.item_rappresentation.predict([X[:,0], X[:,1]])
		profilo_risorsa_dim = [i[0] for i in profilo_risorsa]
		profilo_utente = self.user_rappresentation.predict([np.array([X[0][0]]), np.array([X[0][1]])])[0][0]

		###valutazioni_risorse = model_c.predict([X[:,0], X[:,1]])

		### array di valori di similarita tra il profilo utente e profilo di ogni risorsa 
		return np.array([element[0] for element in cosine_similarity(profilo_risorsa_dim, np.array([profilo_utente]))])#, valutazioni_risorse
		
	### n numero di raccomandazioni richieste 
	def recommender_nn(self, values_sim, n):
		mapping = self.resource_map_encode_reversed
		similarity_sort = values_sim.argsort()[::-1]
		similarity_sort = similarity_sort[1:n+1]
		#data sono gli ID veri delle risorse
		data = [mapping[i] for i in similarity_sort]
		similarity = np.sort(values_sim)[::-1]
		similarity = similarity[1:n+1]
		dictionary = dict(zip(map(str, data), map(float,similarity)))
		#####################################
		###          ratings              ###
		#ratings_final = [rating_risorse[i] for i in similarity_sort]
		#####################################
	
		return dictionary#, ratings_final
		
	@safe_run
	def training_pipeline(self):
		self._dbConnect()
		query = f'SELECT userid,resource,rating FROM mdl_block_rate_resource'
		self._fetchData(query)
		self._dbDisconnect()
		self._cf_training() #collaborative filtering su valutazioni


	############### Training RS - Content Based ###############
	def _preprocessing(self):
		items = self.data_keywords
		low_case_item = items['keywords'].apply(lambda x : x.lower())
		remove_numbers = low_case_item.apply(lambda x : re.sub(r'\d+', '', x))
		replace_chars = remove_numbers.apply(lambda x : x.replace(',' , ' '))
		replace_chars = replace_chars.apply(lambda x : x.replace('-' , ' '))
		remove_punctuation = replace_chars.apply(lambda x : x.translate(str.maketrans('', '', string.punctuation)))
		remove_whitespaces = remove_punctuation.apply(lambda x : x.strip())
		tokenizer = RegexpTokenizer('\w+|\$[\d]+|\w+')
		tokenization = remove_whitespaces.apply(lambda x : tokenizer.tokenize(x))
		stop_words = set(stopwords.words('english')) #todo: add italian
		remove_stop_words = tokenization.apply(lambda x : [i for i in x if not i in stop_words])
		lemmatizer=WordNetLemmatizer()
		lemmatizer_word = remove_stop_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
		items['p_keywords'] = lemmatizer_word

	#create matrix weighted with TF IDF
	def _TF_IDF(self):
		data = self.data_keywords
		data['proc'] = data['p_keywords'].apply(lambda x : ' '.join(x))
		tf_idf = TfidfVectorizer()
		tfidf_matrix = tf_idf.fit_transform(data['proc'])
		latent_matrix = pd.DataFrame(tfidf_matrix.toarray() , index = data.index.tolist())
		return latent_matrix

	def _get_svd_embeddings(self, feature_matrix, n):
		svd = TruncatedSVD(n_components=n, random_state=42, n_iter=7)
		latent_matrix = svd.fit_transform(feature_matrix)
		latent_df = pd.DataFrame(latent_matrix, index = feature_matrix.index.tolist())
		return latent_df

	#restituisce l'ID delle risorse
	def _recommender_cb(self, matrix, n):
		similarity = cosine_similarity(matrix, matrix[-1:]).squeeze(1)
		similarity_sort = similarity.argsort()[::-1]
		similarity_sort = similarity_sort[1:n + 1]

		###view keywords###
		#print(self.data_keywords[self.data_keywords.index.isin(similarity_sort)][['Id_resource', 'keywords']])

		data = self.data_keywords.loc[similarity_sort]['Id_resource'].values.astype(np.int32)
		similarity = np.sort(similarity)[::-1] #reverse sort
		similarity = similarity[1:n+1]
		### MinMax normalize
		#similarity = similarity + np.abs(similarity.min())
		#similarity = similarity / np.abs(similarity).max()

		return dict(list(zip(map(str, data), similarity)))

	def _manager_dataframe(self, user):
		#self.data_keywords = self.data_keywords.query("Property == 'keywords'")[['Id_resource','Value']]#.dropna(subset=['Id_resource', 'Value'])
		self.data_keywords = self.data_keywords[['Id_resource','Value']]
		self.data_keywords = self.data_keywords.groupby('Id_resource')['Value'].apply(lambda x: ','.join(x))
		self.data_keywords = pd.DataFrame(self.data_keywords.items(), columns=['Id_resource', 'keywords'])
		self.data_keywords = self.data_keywords.append(user, ignore_index=True)



	@safe_run
	def content_based_pipeline(self, keywords, n=5):
		user_profile = {'Id_resource':-1, 'keywords': ','.join(keywords)}

		### Load data
		self._dbConnect()
		#query = f'SELECT Id_resource,Value,Property FROM mdl_metadata'
		'''query = f'''
		'''SELECT Id_resource, Value
		FROM (
			SELECT Id_resource, group_concat(keywords) as Value
			FROM (
				SELECT distinct Id_resource,Value as keywords
				FROM mdl_metadata
				WHERE Id_resource is not NULL and Property = "keywords"
				ORDER BY Id_resource,keywords) as t1
				GROUP BY Id_resource) as t2
		GROUP BY Value
		ORDER BY Id_resource
		'''

		query = f'SELECT Id_resource,Value,Property FROM mdl_metadata JOIN mdl_course_modules ON mdl_course_modules.id=mdl_metadata.id_resource WHERE mdl_course_modules.module NOT IN ("26","27","32")'

		self._fetchDataCB(query)
		self._dbDisconnect()

		### Insert user profile in dataset keywords
		self._manager_dataframe(user_profile)

		### Preprocessing keywords
		self._preprocessing()

		### create weighted matrix with tf idf
		latent_matrix = self._TF_IDF()

		### Apply SVD over matrix TF-IDF
		# 25 is the size of the vector representation
		nfeat = min(250, latent_matrix.shape[1]-1)
		matrix_svd = self._get_svd_embeddings(latent_matrix, nfeat)

		### call function reccomendation 
		# 5 is number of items to return
		return self._recommender_cb(matrix_svd, n)


	############### Predict RS - Users ###############
	def _collaborative_filtering_recommend(self, userid):
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
		X = [ [X_user_encoded, resource] for resource in X_resources_encoded ]
		X = np.array(X)

		### Predict
		prediction = self.latest_model.predict([X[:,0], X[:,1]])

		### Assign prediction to resources
		prediction = list(zip(X[:,1], prediction[:,0]))

		### Sort by rating predicted
		prediction = sorted(prediction, key=lambda k: k[1], reverse=True)

		### Keet best 5
		prediction = prediction[:5]

		### Just to get back real values
		result = [ [str(self.resource_map_encode_reversed[int(r[0])]), float(r[1])] for r in prediction ]

		return result
	
	################## Predict RS CF - Embedding rappresentation ###################
	def _Embedding_collaborative_filtering_reccommed(self, userid, n=5):
		### Load mappings
		user_map_encode = 0

		### Get resource _not_ yet evaluated by the user (all_ids - evaluated_ids)
		resources_to_exclude = self.data_df[self.data_df.userid == userid]
		if not resources_to_exclude.shape[0]: return dict()

		resources_to_exclude = resources_to_exclude.resource
		resource_to_evaluate = set(self.data_df.resource) - set(resources_to_exclude)

		### Map resources with mapping
		X_user_encoded = self.user_map_encode[int(userid)]
		X_resources_encoded = list(map(lambda r: self.resource_map_encode[r], resource_to_evaluate))

		### Predictable data structure
		X = [ [X_user_encoded, resource] for resource in X_resources_encoded ]
		X = np.array(X)

		### predict and similarity 
		values_similarity = self.Similarity_NN(X)

		### recommend with real values 
		result = self.recommender_nn(values_similarity, n)

		return result

		
	@safe_run
	def collaborative_filtering_pipeline(self, userid, n):
		userid = int(userid)

		### Load data
		self._dbConnect()

		query = f'SELECT userid,resource FROM mdl_block_rate_resource'
		self._fetchData(query)
		self._dbDisconnect()

		### Get recommendation
		#return self._collaborative_filtering_recommend(userid)

		### Get recommendation embedding
		return self._Embedding_collaborative_filtering_reccommed(userid, n)
		

### Todo:
# at loading time, load model (beware of trainings/day changes)
# add api key for security rezzonz
# exaustive loggings
# comments ;D

def makeSanicResponse(resp):
	print(now(), resp)
	return sanicjson(resp, headers={'Access-Control-Allow-Origin': '*'}, content_type="application/json")

if __name__ == '__main__':
	rs = RS()
	app = Sanic('STRS')

	@app.route('/api/rs', methods=['POST'])
	async def main(request):

		req_parameters = request.json
		print('\n\n', '*'*50)
		print('- request received:', req_parameters)
		print()

		try:

			########## Ping ##########
			if req_parameters['type'] == 'ping':
				return makeSanicResponse({ 'response': 'I recommend to use a recommender' })

			########## Training ##########
			if req_parameters['type'] == 'training':
				rs.training_pipeline()
				return makeSanicResponse({ 'response': 'Training completed' })

			########## Call RS ##########
			if req_parameters['type'] == 'recommend':

				###recommendation_cf = rs.collaborative_filtering_pipeline(req_parameters['userid'], 5)
				#print('recommendation_cf', recommendation_cf)

				recommendation_cf = False #We just want CB
				if bool(recommendation_cf):
					recommendation_cb = rs.content_based_pipeline(req_parameters['keywords'], 5)
					final_recommendation = {**recommendation_cb, **recommendation_cf}
				else:
					recommendation_cb = rs.content_based_pipeline(req_parameters['keywords'], 6) #10)
					final_recommendation = recommendation_cb
					#print('recommendation_cb', recommendation_cb)

				final_recommendation = sorted(list(final_recommendation.items()), key=lambda k: k[1], reverse =True)

				return makeSanicResponse({ 'response': dict(final_recommendation)})

		except Exception as e:
			makeSanicResponse({ 'response': 'Invalid request', 'error': str(e) })


	if __name__ == '__main__':
		### Run flask container
		app.run(
			host='0.0.0.0',
			debug=False,
			port=5001,
			#workers=4,
			#threaded=True,
			#processes=5,

			### !!!!!!!! Renewing certificates:
			### sudo certbot --nginx -d socialthings-rs-ml.westeurope.cloudapp.azure.com
			## cd /etc/letsencrypt/live/socialthings-rs-ml.westeurope.cloudapp.azure.com
			## cp cert.pem privkey.pem /home/rs-whoteach/RS_WT-2020/rs/certificates
			ssl={'cert': 'certificates/cert.pem', 'key': 'certificates/privkey.pem'}
		)

