from numpy import reshape,shape,array,argmax
from numpy.random import randint
import string
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint

# functions to load,clean and tokenize Alice in Wonderland text

def load_text(filename):
	file = open(filename, 'r')
	raw_text = file.read()
	file.close()
	return raw_text


def clean_txt(text):
	clean_list = list()
	cleaner_list = list()
	table = str.maketrans('','', string.punctuation)	
	txt_list = text.split('\n')
	for line in txt_list:
		token = line.split()
		token =  [word.lower() for word in token]
		token = [w.translate(table) for w in token]
		token = [word for word in token if len(word)>1]
		token = [word for word in token if word.isalpha()]
		clean_list.append(token)
	[cleaner_list.append(line) for line in clean_list if line != []]
	return cleaner_list
		

def tokenize(c_list):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(c_list)
	return tokenizer

# function to create X,Y array of train and test sequences

def create_sequences(txt_lst, vocab_size,max_length, tokenizer):
	lst_X, lst_Y = list(), list()
	for i in txt_lst:
		seq = tokenizer.texts_to_sequences([i])[0]
		for j in range(1, len(seq)):
			in_seq, out_seq = seq[:j], seq[j]
			in_seq = pad_sequences([in_seq], maxlen=max_length)
			out_seq = to_categorical([out_seq], num_classes = vocab_size)
			lst_X.append(in_seq)
			lst_Y.append(out_seq)
	return array(lst_X), array(lst_Y)
	

filename = '11-0.txt'
raw_txt =  load_text(filename)
cleaned_txt = clean_txt(raw_txt)
tokenizer = tokenize(cleaned_txt)
vocab_size = len(tokenizer.word_index)+1
max_length  = max(len(word) for word in cleaned_txt)
train_size = len(cleaned_txt)-1000
train,test = cleaned_txt[:train_size], cleaned_txt[train_size:]
trainX, trainY = create_sequences(train, vocab_size, max_length, tokenizer)
testX, testY = create_sequences(test, vocab_size, max_length, tokenizer)


#define different models

def LSTM_model(trainX, trainY, testX, testY,vocab_size, max_length):
	trainX = trainX.reshape(-1,16)
	trainY = trainY.reshape(-1,vocab_size)
	testX = testX.reshape(-1,16)
	testY = testY.reshape(-1,vocab_size)
	visible = Input(shape=(max_length,))
	embed = Embedding(vocab_size,100, mask_zero=True)(visible)
	drop1 = Dropout(0.2,input_shape=(16,))(embed)
	encoder1 = LSTM(100)(drop1)
	decoder1 = Dense(100, activation='relu')(encoder1)
	output = Dense(vocab_size,activation='softmax')(decoder1)
	model = Model(inputs=visible,outputs=output)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	filepath="best-LSTM-weights-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	model.fit(trainX, trainY, epochs=10, callbacks=callbacks_list, validation_data=(testX,testY), verbose=0)
	return model, model.layers[1].get_weights()[0]


L_model,L_weights = LSTM_model(trainX, trainY, testX, testY, vocab_size, max_length) 
L_weight_name = 'best-LSTM-weights-10-4.4614.hdf5'
L_model.load_weights(L_weight_name)
L_model.compile(loss='categorical_crossentropy', optimizer='adam')


def CNN_model(trainX, trainY, testX, testY,vocab_size, max_length):
	trainX = trainX.reshape(-1,16)
	trainY = trainY.reshape(-1,vocab_size)
	testX = testX.reshape(-1,16)
	testY = testY.reshape(-1,vocab_size)
	visible = Input(shape=(max_length,))
	embed = Embedding(vocab_size,100)(visible)
	conv1 = Conv1D(filters=100, kernel_size=5, activation='relu')(embed)
	pool1 = MaxPooling1D(pool_size=2)(conv1)
	flat1 = Flatten()(pool1)
	output = Dense(vocab_size,activation='softmax')(flat1)
	model = Model(inputs=visible,outputs=output)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	filepath="best-CNN-weights-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	model.fit(trainX, trainY, epochs=10, callbacks=callbacks_list, validation_data=(testX,testY), verbose=0)
	return model, model.layers[1].get_weights()[0]


C_model, C_weights = CNN_model(trainX, trainY, testX, testY, vocab_size, max_length) 
C_weight_name = 'best-CNN-weights-10-1.4061.hdf5'
C_model.load_weights(C_weight_name)
C_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Generate text from trained and weighted model

L_w_embed = {w:L_weights[idx] for w, idx in tokenizer.word_index.items()}
C_w_embed = {w:C_weights[idx] for w, idx in tokenizer.word_index.items()}

seed = randint(3,len(trainX)-1)
pattern = trainX[seed]
C_result = []

for i in range(250):
	prediction = C_model.predict(pattern,verbose=0)
	predicted_word = ''
	for word,index in tokenizer.word_index.items():
		if index == prediction.all():
			predicted_word = word
			break
	C_result.append(predicted_word)

C_result = ' '.join(C_result)

L_result = []

for i in range(250):
	prediction = L_model.predict(pattern,verbose=0)
	index = argmax(prediction)
	predicted_max = prediction[0][index]
	predicted_word = ''
	for word,index in L_w_embed.items():
		for i in index:
			for j in prediction[0]:
				if abs(i) == abs(j):
					predicted_word = word
					break
	if predicted_word != '':
		L_result.append(predicted_word)
		




