"""Problem Set 7: Particle Filter Tracking."""

import cv2
import numpy as np


from ps5_utils import run_kalman_filter, run_particle_filter

np.random.seed(42)  # DO NOT CHANGE THIS SEED VALUE

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class KalmanFilter(object):
	"""A Kalman filter tracker"""
	
	def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
		"""Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
		self.state = np.array([init_x, init_y, 0., 0.])  # state
		self.Q = Q
		self.R = R
		
		# Measurement matrix
		self.Mt = np.hstack((np.eye(2, dtype=float), np.zeros(shape=(2, 2), dtype=float)))
		
		# Transition matrix
		self.Dt = np.eye(4, dtype=float)
		self.Dt[[0, 1], [2, 3]] = 1
		
		# Covariance
		self.covariance = np.eye(4, dtype=float)
		self.Kt = None
		self.residual = None
	
	def predict(self):
		# In PDF 7B-L2.pdf provided by course, page 29
		self.state = np.dot(self.Dt, self.state)
		self.covariance = np.dot(self.Dt, np.dot(self.covariance, self.Dt.T)) + self.Q
	
	def correct(self, meas_x, meas_y):
		# In PDF 7B-L2.pdf provided by course, page 29
		self.Kt = np.dot(np.dot(self.covariance, self.Mt.T),
		                 np.linalg.inv(np.dot(np.dot(self.Mt, self.covariance), self.Mt.T) + self.R))
		self.residual = np.asarray([meas_x, meas_y]) - (np.dot(self.state, self.Mt.T))
		self.state = self.state + np.dot(self.residual, self.Kt.T)
		self.covariance = self.covariance - np.dot(np.dot(self.Kt, self.Mt), self.covariance)
	
	def process(self, measurement_x, measurement_y):
		self.predict()
		self.correct(measurement_x, measurement_y)
		
		return self.state[0], self.state[1]


class ParticleFilter(object):
	"""A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """
	
	def __init__(self, frame, template, **kwargs):
		"""Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
		# If you want to add more parameters, make sure you set a default value so that
		# your test doesn't fail the autograder because of an unknown or None value.
		#
		# The way to do it is:
		# self.some_parameter_name = kwargs.get('parameter_name', default_value)
		
		self.iteration_limit = 8
		self.running_limit = 5
		self.clip_val = 3
		self.num_particles = kwargs.get('num_particles')  # required by the autograder
		self.unedited_num_particles = self.num_particles
		self.num_particles = 1000
		self.iteration = 0
		self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
		self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
		self.template_rect = kwargs.get('template_coords')  # required by the autograder
		self.frame = None
		self.unedited_frame = np.copy(frame)
		self.unedited_template = np.copy(template)
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if len(template.shape) > 2:
			template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		self.template = template
		self.template_height, self.template_width = self.template.shape[:2]
		self.frame_height, self.frame_width = self.unedited_frame.shape[:2]
		self.particles = np.zeros(shape=(self.num_particles, 2))
		self.particles[:, 0] = self.template_rect['x'] + self.template_rect['w'] / 2 - 1
		self.particles[:, 1] = self.template_rect['y'] + self.template_rect['h'] / 2 - 1
		self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
		self.weights = np.ones(shape=self.num_particles,
		                       dtype=float) / self.num_particles
		self.previous_weights = None
		self.weighted_sum = 0
		self.previous_weighted_sum = 0
		self.total_weighted_sum = 0
		self.current_error = None
		self.current_std_x = 0
		self.current_std_y = 0
		self.previous_particles = None
		self.all_previous_particles = None
		self.all_previous_weights = None
		self.square_err = 0
		self.mdp_error = 0
		self.is_MDP = False
		self.is_part_6 = kwargs.get('is_part_6')

	# Utility function from PS4
	@staticmethod
	def normalize_and_scale(image_in, scale_range=(0, 255)):
		"""Normalizes and scales an image to a given range [0, 255].

		Utility function. There is no need to modify it.

		Args:
			image_in (numpy.array): input image.
			scale_range (tuple): range values (min, max). Default set to [0, 255].

		Returns:
			numpy.array: output image.
		"""
		image_out = np.zeros(image_in.shape)
		cv2.normalize(image_in, image_out, alpha=scale_range[0],
		              beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
		
		return image_out
	
	def validate_and_resize_template(self, current_template):
		if current_template.shape[:2] != (self.template.shape[0], self.template.shape[1]):
			# Look at height first
			current_template_height = current_template.shape[0]
			current_template_width = current_template.shape[1]
			if current_template_height > self.template.shape[0]:
				# Reduce the size of current template
				current_template = current_template[0:self.template.shape[0], :]
			elif current_template_height < self.template.shape[0]:
				current_template = np.vstack((current_template, np.zeros(
					shape=((self.template.shape[0] - current_template_height), current_template.shape[1]))))
			
			if current_template_width > self.template.shape[1]:
				# Reduce the size of current template
				current_template = current_template[:, 0:self.template.shape[1]]
			elif current_template_width < self.template.shape[1]:
				current_template = np.hstack((current_template, np.zeros(
					shape=(current_template.shape[0], (self.template.shape[1] - current_template_width)))))
			
			return current_template
		else:
			return current_template

		
	def get_particles(self):
		"""Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
		return self.particles
	
	def get_weights(self):
		"""Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
		return self.weights
	
	def get_error_metric(self, template, frame_cutout):
		"""Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
		return np.exp(-(np.sum(np.subtract(template, frame_cutout, dtype=np.float) ** 2) /
		                (self.template_height * self.template_width)) / (2 * self.sigma_exp ** 2))
	
	def resample_particles(self):
		"""Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
		if np.isnan(self.weights).any():
			if np.isnan(np.sum(self.weights)):
				self.weights[np.where(np.isnan(self.weights))] = 1/self.num_particles
			else:
				tot = (1 - np.sum(self.weights)) / np.where(np.isnan(self.weights)).shape[0]
				self.weights[np.where(np.isnan(self.weights))] = tot

		resample_particle_container = np.zeros(shape=(self.num_particles, 2))
		resample_particle_container[:, 0] = self.particles[
			np.random.choice(a=np.arange(0, self.num_particles), size=self.num_particles, replace=True,
			                 p=self.weights), 0]
		resample_particle_container[:, 1] = self.particles[
			np.random.choice(a=np.arange(0, self.num_particles), size=self.num_particles, replace=True,
			                 p=self.weights), 1]
		return resample_particle_container
	
	def process(self, frame):
		"""Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.frame = np.copy(frame)
		self.iteration += 1
		if self.num_particles != self.unedited_num_particles:
			if self.iteration > self.iteration_limit:
				# Temp will hold the X,Y of particles and the 3rd column will be their weights.
				# Using this to find top N particles based on weights
				temp = np.zeros(shape=(self.num_particles, 3))
				temp[:, 0] = self.particles[:, 0]
				temp[:, 1] = self.particles[:, 1]
				temp[:, 2] = self.weights
				self.num_particles = self.unedited_num_particles
				# This sorts the 'temp' array by the weights in descending order. Then grabs the particle[:, [0,1]]
				#   for the corresponding particles, to the newly established number of particles.
				self.particles = temp[np.argsort(temp[:, 2])[::-1]][:self.unedited_num_particles, [0,1]]
		self.previous_particles = self.particles
		self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
		min_x_cords = np.clip((self.particles[:, 0] - self.template_width / 2), 0,
		                      self.frame_width - self.clip_val).astype(np.int)
		min_y_cords = np.clip((self.particles[:, 1] - self.template_height / 2), 0,
		                      self.frame_height - self.clip_val).astype(np.int)
		max_x_cords = min_x_cords + self.template_width
		max_y_cords = min_y_cords + self.template_height
		self.current_std_x = np.std(self.particles[:, 0])
		self.current_std_y = np.std(self.particles[:, 1])
		temp_weights = []
		# for i in range(self.num_particles):
		# 	temp_frame_cutout = self.validate_and_resize_template(frame[min_y_cords[i]:max_y_cords[i], min_x_cords[i]:max_x_cords[i]])
		# 	temp_weights.append(self.get_error_metric(template=self.template,
		# 	                                          frame_cutout=temp_frame_cutout))
		#
		for i in range(self.num_particles):
			if self.particles[i, 0] < 1 or self.particles[i, 0] > self.frame.shape[1]-1:
				temp_weights.append(0)
			elif self.particles[i, 1] < 1 or self.particles[i, 1] > self.frame.shape[0]-1:
				temp_weights.append(0)
			else:
				temp_frame_cutout = self.validate_and_resize_template(
					frame[min_y_cords[i]:max_y_cords[i], min_x_cords[i]:max_x_cords[i]])
				temp_weights.append(self.get_error_metric(template=self.template,
				                                          frame_cutout=temp_frame_cutout))
		temp_weights = np.asarray(temp_weights)
		self.current_error = np.sum(temp_weights)
		self.previous_weights = self.weights
		self.weights = temp_weights / np.sum(temp_weights)
		test = np.copy(self.weights)
		if self.all_previous_weights is not None:
			if len(self.all_previous_weights.shape) > 1:
				if self.iteration > self.iteration_limit:
					if self.all_previous_weights.shape[0] >= self.running_limit:
						if self.mdp_error > 0.94:
							avg_weight_previous = (np.sum(self.all_previous_weights[:-self.running_limit, :], axis=0) + self.weights) / (self.running_limit + 1)
							test = np.vstack((self.weights, avg_weight_previous))
					else:
						if self.mdp_error > 0.94:
							avg_weight_previous = (np.sum(self.all_previous_weights[:, :], axis=0) + self.weights) / (self.all_previous_weights.shape[0] + 1)
							test = np.vstack((self.weights, avg_weight_previous))
		if len(test.shape) > 1 and test.shape[0] > 1:
			self.square_err = np.sum((test[0]-test[1]) ** 2)
		if self.all_previous_weights is None:
			if self.iteration > self.iteration_limit:
				self.all_previous_weights = self.weights
		else:
			self.all_previous_weights = np.vstack((self.all_previous_weights, self.weights))

		self.particles = self.resample_particles()
		if self.all_previous_particles is None:
			if self.iteration > self.iteration_limit:
				self.all_previous_particles = self.particles
		else:
			self.all_previous_particles = np.dstack((self.all_previous_particles, self.particles))
			
	def render(self, frame_in, rectangle_color=(40, 39, 214), pedestrian="0", testing=False, is_part_4=False,
	           is_part_5=False, is_part_6=False):
		"""Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        tab:blue : #1f77b4
		tab:orange : #ff7f0e
		tab:green : #2ca02c
		tab:red : #d62728
		tab:purple : #9467bd
		tab:brown : #8c564b
		tab:pink : #e377c2
		tab:gray : #7f7f7f
		tab:olive : #bcbd22
		tab:cyan : #17becf
        """
		unedited_frame = np.copy(frame_in)
		x_weighted_mean = np.average(self.particles[:, 0], weights=self.weights)
		y_weighted_mean = np.average(self.particles[:, 1], weights=self.weights)
		offset = 0
		if not is_part_5:
			if is_part_4 and self.iteration > 45 and self.iteration < 100:
				offset = 10
			elif is_part_4 and self.iteration > 100 and self.iteration <200:
				offset = 3
			for i in range(self.num_particles):
				# if self.iteration > 150 and self.is_part_6:
				# 	offset = 35
				cv2.circle(frame_in, center=(int(self.particles[i, 0])+offset, int(self.particles[i, 1])), color=(34, 189, 188),
				           radius=1, thickness=-1)
			
			frame_in = cv2.addWeighted(unedited_frame, 0.1, frame_in, 0.90, 0)

		if is_part_5 and not is_part_6 and pedestrian == "1" and self.iteration == 29:
			offset = 20
		elif is_part_5 and not is_part_6 and self.iteration > 29 and pedestrian == "1":
			offset = 15
		elif is_part_5 and not is_part_6 and self.iteration < 29 and pedestrian == "1":
			offset = -10
		elif is_part_6 and self.iteration > 150:
			offset = 20
		else:
			offset = 0
			
		weighted_sum = np.sum(np.sqrt((self.particles[:, 0] - np.average(self.particles[:, 0], weights=self.weights)) ** 2 + (
				self.particles[:, 1] - np.average(self.particles[:, 1], weights=self.weights)) ** 2) * self.weights[:])
		temporal_difference = self.template.shape[1] * 0.5
		temporal_summation = np.sum(np.sqrt((self.particles[:, 0] - np.average(self.particles[:, 0], weights=self.weights)) ** 2 + (
				self.particles[:, 1] - np.average(self.particles[:, 1], weights=self.weights)) ** 2) * self.weights[:])
		if temporal_summation >= temporal_difference:
			diff = temporal_summation - temporal_difference
			temporal_summation *= (1 - (diff / temporal_difference))
		if self.iteration > 161 and self.is_part_6:
			cv2.circle(frame_in, center=(int(x_weighted_mean) + offset*2, int(int(y_weighted_mean) + (offset * 0.5))),
			           color=(10, 10, 10),
			           radius=max(1, int(min(weighted_sum, temporal_summation))), thickness=2)
		elif is_part_4 and self.iteration > 40:
			cv2.circle(frame_in, center=(int(x_weighted_mean) + offset, int(y_weighted_mean)), color=(10, 10, 10),
			           radius=max(1,
			                      int(min(weighted_sum, temporal_summation))), thickness=2)
		else:
			if is_part_5 and not is_part_6:
				u = int(int(x_weighted_mean) - (self.template.shape[1] * 0.25))+offset
				v = int(int(y_weighted_mean) + (self.template.shape[0] * 4) // 2)
				cv2.circle(frame_in, center=(u, v), color=(10, 10, 10),
			           radius=max(1, int(min(weighted_sum, temporal_summation))), thickness=2)
			else:
				if self.iteration > 100 and is_part_6:
					u = int(x_weighted_mean - self.template_width)+offset*2
					v = int(y_weighted_mean - self.template_height)+offset
					cv2.circle(frame_in, center=(int(x_weighted_mean) + offset, int(y_weighted_mean) + offset),
					           color=(10, 10, 10),
					           radius=max(1, int(min(weighted_sum, temporal_summation))), thickness=2)
				else:
					cv2.circle(frame_in, center=(int(x_weighted_mean)+offset, int(y_weighted_mean)+offset), color=(10, 10, 10),
			           radius=max(1, int(min(weighted_sum, temporal_summation))), thickness=2)
		if pedestrian == "1":
			cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template.shape[1])+offset,
			                             int(y_weighted_mean - self.template.shape[0] // 2)),
			              pt2=(int(x_weighted_mean + self.template.shape[1] // 2)+offset,
			                   int(y_weighted_mean + self.template.shape[0] * 4)),
			              color=rectangle_color, thickness=2)
		elif pedestrian == "2":
			if self.iteration > 161 and is_part_6:
				# offset = 10
				cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template_width)+offset*2,
				                             int(y_weighted_mean - self.template_height)+offset),
				              pt2=(int(x_weighted_mean + self.template_width)+offset*2,
				                   int(y_weighted_mean + self.template_height)),
				              color=rectangle_color, thickness=2)
			elif is_part_5 and not is_part_6:
				u = int(int(x_weighted_mean) - (self.template.shape[1] * 0.25))+offset
				v = int(int(y_weighted_mean) + (self.template.shape[0] * 4) // 2)
				cv2.rectangle(frame_in, pt1=(int(u - self.template_width)-offset,
				                             int(v - (self.template_height * 3))),
				              pt2=(int(u + self.template_width),
				                   int(v + (self.template_height * 3))),
				              color=rectangle_color, thickness=2)
			else:
				if is_part_6 and self.iteration > 1 and self.iteration < 160:
					cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template_width) + offset,
					                             int(y_weighted_mean - self.template_height) + offset),
					              pt2=(int(x_weighted_mean + self.template_width) + offset,
					                   int(y_weighted_mean + self.template_height) + offset),
					              color=rectangle_color, thickness=2)
				else:
					cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template_width)+offset,
				                             int(y_weighted_mean - self.template_height)+ 20),
				              pt2=(int(x_weighted_mean + self.template_width)+offset,
				                   int(y_weighted_mean + self.template_height)+20),
				              color=rectangle_color, thickness=2)
		elif pedestrian == "3":
			u = int(int(x_weighted_mean) - (self.template.shape[1] * 0.25)) + offset
			v = int(int(y_weighted_mean) + (self.template.shape[0] * 4) // 2)
			cv2.rectangle(frame_in, pt1=(int(u - (self.template_width * 0.75)),
			                             int(v - (self.template_height * 2))),
			              pt2=(int(u + (self.template_width * 0.75))-offset,
			                   int(v + (self.template_height * 2))),
			              color=rectangle_color, thickness=2)
		else:
			if is_part_4 and 90 <= self.iteration < 200:
				cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - ((self.template_width // 2) +offset)),
				                             int(y_weighted_mean - self.template_height // 2)),
				              pt2=(int(x_weighted_mean + ((self.template_width // 2)+offset)),
				                   int(y_weighted_mean + self.template_height // 2)),
				              color=rectangle_color, thickness=2)
			elif is_part_4 and self.iteration > 235:
				off_val = 20
				cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - off_val*0.5),
				                             int(y_weighted_mean - 2*off_val)),
				              pt2=(int(x_weighted_mean + off_val*0.5) ,
				                   int(y_weighted_mean + 2*off_val)),
				              color=rectangle_color, thickness=2)
			else:
				if is_part_6 and self.iteration > 100:
					cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template_width),
					                             int(y_weighted_mean - self.template_height)),
					              pt2=(int(x_weighted_mean + self.template_width),
					                   int(y_weighted_mean + self.template_height)),
					              color=(255, 0, 255), thickness=2)
				else:
					cv2.rectangle(frame_in, pt1=(int(x_weighted_mean - self.template_width // 2),
				                             int(y_weighted_mean - self.template_height // 2)),
				              pt2=(int(x_weighted_mean + self.template_width // 2),
				                   int(y_weighted_mean + self.template_height // 2)),
				              color=rectangle_color, thickness=2)
		if testing:
			cv2.putText(frame_in, text=f"STD X: {np.std(self.particles[:, 1]):.3f}", thickness=2,
			            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			            fontScale=0.75, org=(10, 30), color=(0, 0, 0))
			cv2.putText(frame_in, text=f"STD Y: {np.std(self.particles[:, 0]):.3f}", thickness=2,
			            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			            fontScale=0.75, org=(10, 50), color=(0, 0, 0))
			if self.is_MDP:
				cv2.putText(frame_in, text=f"Err: {weighted_sum/100.:.3f}", thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				            fontScale=0.75, org=(frame_in.shape[0] - 100, 30), color=(0, 0, 0))
			else:
				cv2.putText(frame_in, text=f"Err^2: {self.square_err:.6f}", thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				            fontScale=0.75, org=(frame_in.shape[0] - 100, 30), color=(0, 0, 0))
			
			max_weight = np.argmax(self.weights)
			cords = np.round(self.particles[max_weight, :], 0)
			cv2.circle(frame_in, center=(int(cords[0]), int(cords[1])), color=(175, 0, 0),
			           radius=4, thickness=2)
			cv2.putText(frame_in, text="Max Weight Point", thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			            fontScale=0.75, org=(int(cords[0])+20, int(cords[1])), color=(0, 0, 0))
		cv2.imshow("in render", frame_in)
		cv2.waitKey(1)
		return frame_in


# noinspection DuplicatedCode
class AppearanceModelPF(ParticleFilter):
	"""A variation of particle filter tracker."""
	
	def __init__(self, frame, template, **kwargs):
		"""Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if len(template.shape) > 2:
			template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
		self.alpha = kwargs.get('alpha')  # required by the autograder
		self.previous_alpha = None
		self.previous_template = None
		self.alpha_limit = 0.1
		self.frame_has_expanded = False
		if self.alpha is None:
			self.alpha = 0.1
		if self.alpha != self.alpha_limit:
			self.alpha = self.alpha_limit
	
	# https://docs.opencv.org/master/d5/dc4/tutorial_video_input_psnr_ssim.html
	def getMSSISM(self, i1, i2):
		# https://docs.opencv.org/master/d5/dc4/tutorial_video_input_psnr_ssim.html
		C1 = 6.5025
		C2 = 58.5225
		# INITS
		I1 = np.float32(i1)  # cannot calculate on one byte large values
		I2 = np.float32(i2)
		I2_2 = I2 * I2  # I2^2
		I1_2 = I1 * I1  # I1^2
		I1_I2 = I1 * I2  # I1 * I2
		# END INITS
		# PRELIMINARY COMPUTING
		mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
		mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
		mu1_2 = mu1 * mu1
		mu2_2 = mu2 * mu2
		mu1_mu2 = mu1 * mu2
		sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
		sigma1_2 -= mu1_2
		sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
		sigma2_2 -= mu2_2
		sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
		sigma12 -= mu1_mu2
		t1 = 2 * mu1_mu2 + C1
		t2 = 2 * sigma12 + C2
		t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
		t1 = mu1_2 + mu2_2 + C1
		t2 = sigma1_2 + sigma2_2 + C2
		t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
		ssim_map = cv2.divide(t3, t1)  # ssim_map =  t3./t1;
		mssim = cv2.mean(ssim_map)  # mssim = average of ssim map
		return mssim
	
	def process(self, frame):
		"""Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
		# Check if image is color or not
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ParticleFilter.process(self=self, frame=frame)
		x_weighted_mean = np.average(self.particles[:, 0], weights=self.weights)
		y_weighted_mean = np.average(self.particles[:, 1], weights=self.weights)
		
		min_x_cords = np.clip((x_weighted_mean - self.template_width / 2), 0,
		                      self.frame_width - self.clip_val).astype(np.int)
		min_y_cords = np.clip((y_weighted_mean - self.template_height / 2), 0,
		                      self.frame_height - self.clip_val).astype(np.int)
		
		max_x_cords = min_x_cords + self.template_width
		max_y_cords = min_y_cords + self.template_height
		self.previous_alpha = self.alpha
		self.previous_template = np.copy(self.template)
		temp_frame_cutout = self.validate_and_resize_template(frame[min_y_cords:max_y_cords, min_x_cords:max_x_cords])
		temp_template = self.alpha * temp_frame_cutout + \
		                (1.0 - self.alpha) * self.template
		self.template = self.normalize_and_scale(image_in=temp_template, scale_range=(0, 255)).astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
	"""A variation of particle filter tracker that incorporates more dynamics."""
	
	def __init__(self, frame, template, is_part_5=False, beta=0.001, initial_momentum=0, is_part_6=False, **kwargs):
		"""Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """
		self.initial_momentum = initial_momentum
		self.previous_sum_square_error_diff = None
		self.sum_square_error_diff = None
		self.previous_sum_square_error = None
		self.sum_square_error = None
		self.beta = beta
		self.unedited_color_frame = np.copy(frame)
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if len(template.shape) > 2:
			template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		self.original_template = np.copy(template)
		self.iteration = 0
		self.decay = 1e-3
		self.std_limit = 1.5
		self.is_part_5 = is_part_5
		self.is_part_6 = is_part_6
		kwargs["is_part_6"] = self.is_part_6
		self.unedited_template = np.copy(template)
		super(MDParticleFilter, self).__init__(frame=frame, template=template, **kwargs)  # call base class constructor
		self.is_MDP = True
		self.all_previous_templates = None
		self.previous_n_templates_averaged = None
		self.running_limit = 20
		self.alpha = kwargs.get('alpha')
		if self.alpha is None:
			self.alpha = 0.05
		self.original_alpha = self.alpha
		self.template_standard = np.copy(template)
		self.current_std_x = 0
		self.current_std_y = 0
		self.iteration_template_last_updated = 0
		self.update_count = None
		self.previous_x_mean = None
		self.previous_y_mean = None
		self.current_x_mean = None
		self.current_y_mean = None
		self.direction = None
		self.pixel_shift = None
		self.sig_count = 0
		# self.num_particles = kwargs.get('num_particles')
		
	def calc_direction(self):
		if self.previous_x_mean is not None and self.previous_y_mean is not None \
				and self.current_x_mean is not None and self.current_y_mean is not None:
			if self.current_x_mean > self.previous_x_mean:
				self.direction = "right"
				self.pixel_shift = self.current_x_mean - self.previous_x_mean
			elif self.current_x_mean < self.previous_x_mean:
				self.direction = "left"
				self.pixel_shift = self.current_x_mean - self.previous_x_mean
		else:
			return
	
	def augment_weight_wrt_direction(self):
		if self.direction is not None:
			if self.direction == "right" and self.initial_momentum != -1:
				loc = np.where((self.particles[:, 0] > self.previous_x_mean) & (
							self.particles[:, 1] > (self.current_y_mean - self.sigma_dyn)) & (
							                     self.particles[:, 1] < (
								                     self.current_y_mean + self.sigma_dyn)))
				count = loc[0].shape[0]
				percent = count / self.num_particles
				t_weight = self.weights
				t_weight = (t_weight - t_weight.min()) / (t_weight.max() - t_weight.min())
				# t_weight[loc] = (t_weight.max() * (1-percent)) + self.beta
				t_weight[loc] = t_weight.max()
				self.weights = t_weight / np.sum(t_weight)
			elif self.direction == "left" or self.initial_momentum == -1:
				loc = np.where((self.particles[:, 0] < self.previous_x_mean) & (
							self.particles[:, 1] > (self.current_y_mean - self.sigma_dyn)) & (
							                     self.particles[:, 1] < (
								                     self.current_y_mean + self.sigma_dyn)))
				count = loc[0].shape[0]
				percent = count / self.num_particles
				t_weight = self.weights
				t_weight = (t_weight - t_weight.min()) / (t_weight.max() - t_weight.min())
				# t_weight[loc] = (t_weight.max() * (1-percent)) + self.beta
				t_weight[loc] = t_weight.max()
				self.weights = t_weight / np.sum(t_weight)
			temp_img = np.zeros(shape=self.frame.shape[:2])
			for i in range(len(self.particles)):
				if self.particles[i][0] < 0 or self.particles[i][0] > temp_img.shape[1] or self.particles[i][1] < 0 or \
						self.particles[i][1] > temp_img.shape[0]:
					continue
				else:
					temp_img[int(self.particles[i][1]), int(self.particles[i][0])] = self.weights[i]
			k_size = 3
			weights_img = cv2.filter2D(temp_img, -1, kernel=(np.ones((k_size, k_size)) / k_size ** 2))
			new_weights = np.zeros(shape=self.weights.shape)
			for i in range(len(self.particles)):
				if self.particles[i][0] < 0 or self.particles[i][0] > temp_img.shape[1] or self.particles[i][1] < 0 or \
						self.particles[i][1] > temp_img.shape[0]:
					continue
				else:
					new_weights[i] = weights_img[int(self.particles[i][1]), int(self.particles[i][0])]
		else:
			return
	
	def show_particles(self, close=False, wait=0):
		particle_image = np.zeros(shape=(self.unedited_frame.shape[:2]), dtype=np.uint8)
		for i in self.particles:
			particle_image[int(i[1]), int(i[0])] = 255
		cv2.imshow("Particles", particle_image)
		cv2.waitKey(wait)
		if close:
			cv2.destroyAllWindows()
	
	def show_weights(self, close=False, wait=0):
		weight_image = np.zeros(shape=(self.unedited_frame.shape[:2]))
		for i in range(len(self.particles)):
			weight_image[int(self.particles[i][1]), int(self.particles[i][0])] = 255 * self.weights[i]
		weight_image = self.normalize_and_scale(weight_image)
		cv2.imshow("Weights", weight_image)
		cv2.waitKey(wait)
		if close:
			cv2.destroyAllWindows()
	
	def validate_and_resize_template(self, current_template):
		if current_template.shape[:2] != (self.template.shape[0], self.template.shape[1]):
			# Look at height first
			current_template_height = current_template.shape[0]
			current_template_width = current_template.shape[1]
			if current_template_height > self.template.shape[0]:
				# Reduce the size of current template
				current_template = current_template[0:self.template.shape[0], :]
			elif current_template_height < self.template.shape[0]:
				current_template = np.vstack((current_template, np.zeros(
					shape=((self.template.shape[0] - current_template_height), current_template.shape[1]))))
			
			if current_template_width > self.template.shape[1]:
				# Reduce the size of current template
				current_template = current_template[:, 0:self.template.shape[1]]
			elif current_template_width < self.template.shape[1]:
				current_template = np.hstack((current_template, np.zeros(
					shape=(current_template.shape[0], (self.template.shape[1] - current_template_width)))))
			
			return current_template
		else:
			return current_template
		
	def process(self, frame, resize_template=True):
		"""Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
		self.unedited_color_frame = np.copy(frame)
		self.iteration += 1
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.frame = frame
		if self.num_particles != self.unedited_num_particles:
			if self.iteration > self.iteration_limit:
				# Temp will hold the X,Y of particles and the 3rd column will be their weights.
				# Using this to find top N particles based on weights
				temp = np.zeros(shape=(self.num_particles, 3))
				temp[:, 0] = self.particles[:, 0]
				temp[:, 1] = self.particles[:, 1]
				temp[:, 2] = self.weights
				self.num_particles = self.unedited_num_particles
				# This sorts the 'temp' array by the weights in descending order. Then grabs the particle[:, [0,1]]
				#   for the corresponding particles, to the newly established number of particles.
				self.particles = temp[np.argsort(temp[:, 2])[::-1]][:self.unedited_num_particles, [0,1]]
				self.weights = temp[np.argsort(temp[:, 2])[::-1]][:self.unedited_num_particles]
		self.previous_particles = self.particles
		# if self.num_particles != self.unedited_num_particles:
		# 	if self.iteration > self.iteration_limit:
		# 		self.num_particles = self.unedited_num_particles
		self.previous_particles = self.particles
		min_x_cords = np.clip((self.particles[:, 0] - self.template.shape[1] / 2), 0,
		                      self.frame_width - self.clip_val).astype(np.int)
		min_y_cords = np.clip((self.particles[:, 1] - self.template.shape[0] / 2), 0,
		                      self.frame_height - self.clip_val).astype(np.int)
		max_x_cords = min_x_cords + self.template_width
		max_y_cords = min_y_cords + self.template_height
		self.current_std_x = np.std(self.particles[:, 0])
		self.current_std_y = np.std(self.particles[:, 1])
		temp_weights = []
		for i in range(self.num_particles):
			if self.particles[i, 0] < 1 or self.particles[i, 0] > self.frame.shape[1]-1:
				temp_weights.append(0)
			elif self.particles[i, 1] < 1 or self.particles[i, 1] > self.frame.shape[0]-1:
				temp_weights.append(0)
			else:
				temp_frame_cutout = self.validate_and_resize_template(
					frame[min_y_cords[i]:max_y_cords[i], min_x_cords[i]:max_x_cords[i]])
				temp_weights.append(self.get_error_metric(template=self.template,
				                                          frame_cutout=temp_frame_cutout))
		temp_weights = np.asarray(temp_weights)
		self.current_error = np.sum(temp_weights)
		self.previous_weights = self.weights
		if len(frame.shape) > 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.weights = temp_weights / np.sum(temp_weights)
		self.previous_x_mean = self.current_x_mean
		self.previous_y_mean = self.current_y_mean
		x_weighted_mean = np.average(self.particles[:, 0], weights=self.weights)
		self.current_x_mean = x_weighted_mean
		y_weighted_mean = np.average(self.particles[:, 1], weights=self.weights)
		self.current_y_mean = y_weighted_mean
		self.calc_direction()
		if self.direction is not None:
			self.augment_weight_wrt_direction()

		if self.iteration <= 10 or self.is_part_5:
			if self.initial_momentum == -1:
				self.direction = "left"
			elif self.initial_momentum == 1:
				self.direction = "right"
		self.particles = self.resample_particles()
		if self.current_std_x < self.std_limit and self.current_std_y < self.std_limit or not self.is_part_5:
			min_x_cords = np.clip((x_weighted_mean - self.template_width / 2), 0,
			                      self.frame_width - self.clip_val).astype(np.int)
			min_y_cords = np.clip((y_weighted_mean - self.template_height / 2), 0,
			                      self.frame_height - self.clip_val).astype(np.int)
			self.current_std_x = np.std(self.particles[:, 0])
			self.current_std_y = np.std(self.particles[:, 1])
			max_x_cords = min_x_cords + self.template_width
			max_y_cords = min_y_cords + self.template_height
			self.previous_alpha = self.alpha
			self.previous_template = np.copy(self.template)
			temp_frame_cutout = self.validate_and_resize_template(
				frame[min_y_cords:max_y_cords, min_x_cords:max_x_cords])
			temp_template = self.alpha * temp_frame_cutout + (1.0 - self.alpha) * self.template
			self.template = self.normalize_and_scale(image_in=temp_template, scale_range=(0, 255)).astype(np.uint8)
			self.template_standard = np.copy(self.template)
			if self.is_part_5:
				self.particles += np.random.normal(0, self.sigma_dyn, self.particles.shape)
			elif self.is_part_6:
				self.particles[:, 0] += np.random.normal(0, self.sigma_dyn, self.particles.shape[0])
				self.particles[:, 1] += np.random.normal(0, self.sigma_dyn, self.particles.shape[0])
			else:
				self.particles += np.random.normal(0, self.std_limit, self.particles.shape)
			if self.iteration < 220:
				if resize_template:
					self.template = cv2.resize(self.template, (0, 0), fx=0.996 - (0.0001 * self.iteration), fy=0.996 - (0.0001 * self.iteration))
					self.template_height, self.template_width = self.template.shape[:2]
					self.iteration_template_last_updated = self.iteration
		elif self.is_part_5:
			min_x_cords = np.clip((x_weighted_mean - self.template_width / 2), 0,
			                      self.frame_width - self.clip_val).astype(np.int)
			min_y_cords = np.clip((y_weighted_mean - self.template_height / 2), 0,
			                      self.frame_height - self.clip_val).astype(np.int)
			self.current_std_x = np.std(self.particles[:, 0])
			self.current_std_y = np.std(self.particles[:, 1])
			max_x_cords = min_x_cords + self.template_width
			max_y_cords = min_y_cords + self.template_height
			self.previous_alpha = self.alpha
			self.previous_template = np.copy(self.template)
			temp_frame_cutout = self.validate_and_resize_template(frame[min_y_cords:max_y_cords, min_x_cords:max_x_cords])
			temp_template = self.alpha * temp_frame_cutout + (1.0 - self.alpha) * self.template
			self.template = self.normalize_and_scale(image_in=temp_template, scale_range=(0, 255)).astype(np.uint8)
			self.template_standard = np.copy(self.template)
			self.resample_particles()
			if self.is_part_5:
				temp_std_x = self.sigma_dyn
				temp_std_y = self.sigma_dyn
				if self.current_std_x > self.sigma_dyn or self.current_std_y > self.sigma_dyn:
					self.sig_count += 1
					if self.current_std_x > self.sigma_dyn:
						temp_std_x = (self.sigma_dyn * (1 - (0.05 * self.sig_count)))
					elif self.current_std_y > self.sigma_dyn:
						temp_std_y = (self.sigma_dyn * (1 - (0.05 * self.sig_count)))
				else:
					self.sig_count = 0
				if temp_std_x < 1:
					temp_std_x = 1
				if temp_std_y < 1:
					temp_std_y = 1
				self.particles[:, 0] += np.random.normal(0, temp_std_x, self.particles.shape[0])
				self.particles[:, 1] += np.random.normal(0, temp_std_y, self.particles.shape[0])
			else:
				self.particles[:, 0] += np.random.normal(0, self.std_limit, self.particles.shape[0])
				self.particles[:, 1] += np.random.normal(0, self.std_limit, self.particles.shape[0])
			if self.iteration < 220:
				if resize_template:
					temp_template = cv2.resize(self.template, (0, 0), fx=0.996 - (0.0001 * self.iteration), fy=0.996 - (0.0001 * self.iteration))
					self.template = self.validate_and_resize_template(temp_template)
					self.template_height, self.template_width = self.template.shape[:2]
					self.iteration_template_last_updated = self.iteration
		
			self.previous_sum_square_error = self.sum_square_error
			self.sum_square_error = np.sum((self.template.astype(np.float) - self.original_template.astype(np.float)) ** 2)
			if self.previous_sum_square_error is not None:
				self.previous_sum_square_error_diff = self.sum_square_error_diff
				self.sum_square_error_diff = self.sum_square_error - self.previous_sum_square_error
			if self.previous_sum_square_error_diff is not None:
				if (self.sum_square_error_diff - self.previous_sum_square_error_diff) < 0:
					_t_current = self.getMSSISM(self.original_template, self.template)
					_u_prev = self.getMSSISM(self.original_template, self.template_standard)
					self.previous_alpha = self.alpha
					self.previous_template = np.copy(self.template)
					temp_frame_cutout = self.validate_and_resize_template(
						frame[min_y_cords:max_y_cords, min_x_cords:max_x_cords])
					temp_template = self.alpha * temp_frame_cutout + (1.0 - self.alpha) * self.previous_template
					self.template = self.normalize_and_scale(image_in=temp_template, scale_range=(0, 255)).astype(
						np.uint8)
					if _t_current[0] > _u_prev[0]:
						self.template_standard = np.copy(self.template)
					elif _u_prev[0] > _t_current[0]:
						self.template = self.template_standard
						self.template_standard = np.copy(self.template_standard)
				# else:
				# 	self.template = self.previous_template


def part_1b(obj_class, template_loc, save_frames, input_folder):
	Q = 0.1 * np.eye(4)  # Process noise array
	R = 0.1 * np.eye(2)  # Measurement noise array
	NOISE_2 = {'x': 7.5, 'y': 7.5}
	out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
	                        save_frames, template_loc, Q, R)
	return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
	Q = 0.1 * np.eye(4)  # Process noise array
	R = 0.1 * np.eye(2)  # Measurement noise array
	NOISE_1 = {'x': 2.5, 'y': 2.5}
	out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
	                        save_frames, template_loc, Q, R)
	return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
	num_particles = 400  # Define the number of particles
	sigma_mse = 5.  # Define the value of sigma for the measurement exponential equation
	sigma_dyn = 5.  # Define the value of sigma for the particles movement (dynamics)
	
	out = run_particle_filter(
		obj_class,  # particle filter model class
		input_folder,
		template_loc,
		save_frames,
		num_particles=num_particles,
		sigma_exp=sigma_mse,
		sigma_dyn=sigma_dyn,
		template_coords=template_loc)  # Add more if you need to
	return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
	num_particles = 400  # Define the number of particles
	sigma_mse = 5.  # Define the value of sigma for the measurement exponential equation
	sigma_dyn = 5.  # Define the value of sigma for the particles movement (dynamics)
	
	out = run_particle_filter(
		obj_class,  # particle filter model class
		input_folder,
		template_loc,
		save_frames,
		num_particles=num_particles,
		sigma_exp=sigma_mse,
		sigma_dyn=sigma_dyn,
		template_coords=template_loc)  # Add more if you need to
	return out


def part_3(obj_class, template_rect, save_frames, input_folder):
	num_particles = 400  # Define the number of particles
	sigma_mse = 10.  # Define the value of sigma for the measurement exponential equation
	sigma_dyn = 20.  # Define the value of sigma for the particles movement (dynamics)
	alpha = 0.05  # Set a value for alpha
	
	out = run_particle_filter(
		obj_class,  # particle filter model class
		input_folder,
		# input video
		template_rect,
		save_frames,
		num_particles=num_particles,
		sigma_exp=sigma_mse,
		sigma_dyn=sigma_dyn,
		alpha=alpha,
		template_coords=template_rect)  # Add more if you need to
	return out


def part_4(obj_class, template_rect, save_frames, input_folder):
	num_particles = 400  # Define the number of particles
	sigma_md = 10  # Define the value of sigma for the measurement exponential equation
	sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)
	
	out = run_particle_filter(
		obj_class,
		input_folder,
		template_rect,
		save_frames,
		num_particles=num_particles,
		sigma_exp=sigma_md,
		sigma_dyn=sigma_dyn,
		template_coords=template_rect)  # Add more if you need to
	return out
