import numpy as np
import itertools

class MonsterDiagnosisAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        self.diseases = {}
        self.disease_index = None
        self.disease_array = None
        self.disease_array_ii = None
        self.patient = {}
        self.patient_array = None
        self.patient_array_ii = None
        pass
    
    def determine_required_remaining(self):
        return
    
    def patient_to_ii(self, patient_array):
        self.patient_array_ii = np.clip(np.cumsum(patient_array, axis=0), -1, 1)
        return
    
    def diseases_to_ii(self, diseases_array, return_val=False):
        if return_val:
            if len(diseases_array) > 1:
                return np.clip(np.sum(diseases_array, axis=0), -1, 1)
            else:
                return diseases_array
        else:
            self.disease_array_ii = np.clip(np.sum(np.cumsum(diseases_array, axis=0), axis=0), -1, 1)
            return
        
    def diseases_to_array(self, diseases, set_index=False):
        temp_diseases = np.zeros(shape=(len(diseases), len(diseases[next(iter(diseases))])))
        self.disease_index = {}
        if set_index:
            self.index_to_disease = {}
        
        count = 0
        for key, val in diseases.items():
            if key not in self.disease_index:
                self.disease_index[key] = count
                if set_index:
                    self.index_to_disease[count] = key
            temp_diseases[count, :] = self.to_array(val=val)
            count += 1
        self.disease_array = temp_diseases
    
    def to_array(self, val):
        temp_result = np.zeros(shape=(len(val), ))
        count = 0
        for k, v in val.items():
            if v == "+":
                temp_result[count] = 1
            elif v == "-":
                temp_result[count] = -1
            else:
                temp_result[count] = 0
            count += 1
        return temp_result
    
    def get_val_from_tuple(self, tup, col):
        val = np.sum([self.disease_array[i, col] for i in tup])
        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        return val
    
    def setup(self, diseases, patient):
        self.diseases = diseases
        self.patient = patient
        self.patient_array = self.to_array(patient)
        self.diseases_to_array(diseases=diseases, set_index=True)
        self.diseases_to_ii(diseases_array=self.disease_array, return_val=False)
        self.patient_to_ii(patient_array=self.patient_array)
        return

    def solve(self, diseases, patient):
        # Add your code here!
        #
        # The first parameter to this method is a list of diseases, represented as a
        # list of 2-tuples. The first item in each 2-tuple is the name of a disease. The
        # second item in each 2-tuple is a dictionary of symptoms of that disease, where
        # the keys are letters representing vitamin names ("A" through "Z") and the values
        # are "+" (for elevated), "-" (for reduced), or "0" (for normal).
        #
        # The second parameter to this method is a particular patient's symptoms, again
        # represented as a dictionary where the keys are letters and the values are
        # "+", "-", or "0".
        #
        # This method should return a list of names of diseases that together explain the
        # observed symptoms. If multiple lists of diseases can explain the symptoms, you
        # should return the smallest list. If multiple smallest lists are possible, you
        # may return any sufficiently explanatory list.

        self.setup(diseases=diseases, patient=patient)
        # self.disease_array = np.vstack((self.disease_array, self.patient_array))

        if np.any(np.all(self.patient_array == self.disease_array, axis=1)):
            return self.index_to_disease[np.argwhere(
                np.all(self.patient_array == self.disease_array, axis=1))[0]]

        possible_diseases = [key for key, _ in diseases.items()]
        absolutely_required = []
        for i in range(len(self.patient_array)):
            patient_value = self.patient_array[i]
            if patient_value == 1:
                if 1 in self.disease_array[:, i]:
                    idx = np.where(self.disease_array[:, i] == 1)
                    if len(idx) == 1:
                        if len(idx[0]) == 1:
                            if idx[0][0] not in absolutely_required:
                                absolutely_required.append(idx[0][0])
            elif patient_value == -1:
                if -1 in self.disease_array[:, i]:
                    idx = np.where(self.disease_array[:, i] == -1)
                    if len(idx) == 1:
                        if len(idx[0]) == 1:
                            if idx[0][0] not in absolutely_required:
                                absolutely_required.append(idx[0][0])

        disease_vals = self.index_to_disease.keys()
        if len(disease_vals) > 15:
            for i in range(min(6, len(disease_vals))):
                temp = list(itertools.combinations(self.index_to_disease.keys(), i+1))
                for j in temp:
                    temp_result = self.diseases_to_ii(diseases_array=self.disease_array[j, :], return_val=True)
                    if len(temp_result.shape) == 2:
                        check = np.all(temp_result[-1, :] == self.patient_array)
                        if check:
                            return [self.index_to_disease[i] for i in j]
                    else:
                        check = np.all(temp_result == self.patient_array)
                        if check:
                            return [self.index_to_disease[i] for i in j]
        else:
            pos = []
            for i in range(min(10, len(disease_vals))):
                temp = list(itertools.combinations(self.index_to_disease.keys(), i + 1))
                for j in temp:
                    temp_result = self.diseases_to_ii(diseases_array=self.disease_array[j, :], return_val=True)
                    if len(temp_result.shape) == 2:
                        check = np.all(temp_result[-1, :] == self.patient_array)
                        if check:
                            return [self.index_to_disease[i] for i in j]
                    else:
                        check = np.all(temp_result == self.patient_array)
                        if check:
                            return [self.index_to_disease[i] for i in j]
        return [self.index_to_disease[i] for i in range(len(self.index_to_disease))]


