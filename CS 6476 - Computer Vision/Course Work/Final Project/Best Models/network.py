from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.utils

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')


class ConvolutionalNeuralNetwork:
	def __init__(self, input_shape=(16, 32, 3), output_size=10, model_type="LeNet5", learning_rate=0.001):
		self.model = None
		self.learning_rate = learning_rate
		self.input_shape = input_shape
		self.output_size = output_size
		self.model_type = model_type
		self.setup()
		
	def predict(self):
		return
	
	def train(self):
		return
	
	def evaluate(self):
		return
	
	def create_convolutional_segment(self, X, mid_conv_window, filters, name, name_pt_2, stride=2, add_dropout=False):
		convolutional_layer_base = 'res' + str(name) + name_pt_2 + '_branch'
		batch_norm_layer_base = 'bn' + str(name) + name_pt_2 + '_branch'
		F1, F2, F3 = filters
		X_shortcut = X
		X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride), name=convolutional_layer_base + '2a',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2a')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F2, kernel_size=(mid_conv_window, mid_conv_window), strides=(1, 1), name=convolutional_layer_base + '2b', padding='same',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2b')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=convolutional_layer_base + '2c', padding='valid',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2c')(X)
		X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(stride, stride), name=convolutional_layer_base + '1', padding='valid',
		                    kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
		X_shortcut = BatchNormalization(axis=3, name=batch_norm_layer_base + '1')(X_shortcut)
		X = tensorflow.keras.layers.Add()([X, X_shortcut])
		X = Activation('relu')(X)
		return X

	def create_identity_segment(self, X, mid_conv_window, filters, name, name_pt_2):
		conv_name_base = 'res' + str(name) + name_pt_2 + '_branch'
		bn_name_base = 'bn' + str(name) + name_pt_2 + '_branch'
		F1, F2, F3 = filters
		X_shortcut = X

		X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F2, kernel_size=(mid_conv_window, mid_conv_window),
		           strides=(1, 1), padding='same', name=conv_name_base + '2b',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

		X = tensorflow.keras.layers.Add()([X_shortcut, X])
		X = Activation('relu')(X)
		return X
	
	def create_model(self, input_shape, model_type="LeNet5"):
		if model_type == "LeNet5" or model_type.lower() == "lenet5" or model_type.lower() == "lenet":
			X_in = Input(input_shape)

			X = ZeroPadding2D((3, 3))(X_in)
			
			num_filters_conv1 = 32
			X = Conv2D(num_filters_conv1, (5, 5), strides=(1, 1), name='convolution_zero')(X)
			X = BatchNormalization(axis=3, name='batch_normalization_zero')(X)
			X = AveragePooling2D((2, 2), name='max_pool_zero')(X)
			X = Activation('relu')(X)
			filter_1 = [64, 64, 256]
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='a', stride=1)
			X = Dropout(0.2)(X)

			num_filters_conv2 = 32
			X = Conv2D(num_filters_conv2, (3, 3), strides=(1, 1), name='convolution_one')(X)
			X = BatchNormalization(axis=3, name='batch_normalization_one')(X)
			X = AveragePooling2D((2, 2), name='max_pool_one')(X)
			X = Activation('relu')(X)
			
			X = Flatten()(X)
			X = Dense(1048, activation='sigmoid', name='fully_connected_zero')(X)
			X = Dropout(0.5)(X)

			X = Flatten()(X)
			X = Dense(512, activation='sigmoid', name='fully_connected_one')(X)
			X = Dropout(0.5)(X)

			X = Flatten()(X)
			X = Dense(64, activation='sigmoid', name='fully_connected_three')(X)

			X = Flatten()(X)
			X = Dense(self.output_size, activation='softmax', name='fully_connected_four')(X)
			
			model = Model(inputs=X_in, outputs=X, name='LeNet5')
			return model
		elif model_type == "VGG16" or model_type.lower() == "vgg16":
			vgg = tensorflow.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
			for layer in vgg.layers[:-1]:
				layer.trainable = True
			X = Flatten()(vgg.output)
			# filter_1 = [64, 64, 256]
			# X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1,
			#                                       name=2, name_pt_2='a', stride=1, add_dropout=True)
			X = Dense(self.output_size, activation='softmax')(X)
			model = Model(inputs=vgg.input, outputs=X)
			return model
		elif model_type == "ResNet" or model_type.lower() == "resnet":
			X_in = Input(input_shape)
			X = ZeroPadding2D((3, 3))(X_in)

			X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			X = BatchNormalization(axis=3, name='bn_conv1')(X)
			X = Activation('relu')(X)
			X = MaxPooling2D((3, 3), strides=(2, 2))(X)
			
			filter_1 = [64, 64, 256]
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='a', stride=1)
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='b', stride=1)
			X = self.create_identity_segment(X, 3, filter_1, name=2, name_pt_2='c')
			X = self.create_identity_segment(X, 3, filter_1, name=2, name_pt_2='d')
			
			filter_2 = [128, 128, 512]
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_2, name=3, name_pt_2='a', stride=2)
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='b')
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='c')
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='d')
			
			# filter_3 = [256, 256, 1024]
			# X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_3, name=4, name_pt_2='a', stride=2)
			# X = self.create_identity_segment(X, 3, filter_3, name=4, name_pt_2='b')
			# X = self.create_identity_segment(X, 3, filter_3, name=4, name_pt_2='c')
			# X = self.create_identity_segment(X, 3, filter_3, name=4, name_pt_2='d')
			# X = self.create_identity_segment(X, 3, filter_3, name=4, name_pt_2='e')
			# X = self.create_identity_segment(X, 3, filter_3, name=4, name_pt_2='f')
			
			#
			# X = self.create_convolutional_segment(X, mid_conv_window=3, filters=[512, 512, 2048], name=5, name_pt_2='a', stride=2)
			# X = self.create_identity_segment(X, 3, [512, 512, 2048], name=5, name_pt_2='b')
			# X = self.create_identity_segment(X, 3, [512, 512, 2048], name=5, name_pt_2='c')

			X = AveragePooling2D((2, 2), name="avg_pool")(X)

			X = Flatten()(X)
			X = Dense(self.output_size, activation='softmax', name='fc' + str(self.output_size),
			          kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)

			model = Model(inputs=X_in, outputs=X, name='ResNet50')
			return model
	
	def setup(self):
		self.model = self.create_model(self.input_shape, model_type=self.model_type)
		self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate),
		                   loss=tensorflow.keras.losses.categorical_crossentropy, metrics=["accuracy"])
		return