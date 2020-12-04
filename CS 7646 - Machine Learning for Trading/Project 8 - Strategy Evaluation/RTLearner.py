# Import allowed per Steven B. Bryant https://piazza.com/class/kdthusf8jeo7ia?cid=109_f2
import DTLearner


class RTLearner(DTLearner.DTLearner):
    def __init__(self, leaf_size=5, verbose=False, **kwargs):
        # region Found on Stackoverflow
        self.verbose = verbose
        self.leaf_size = leaf_size
        allowed_keys = {'leaf_size', 'verbose'}
        for key, val in kwargs.items():
            if key == 'kwargs':
                kwargs = kwargs["kwargs"]
                break
        for key, val in kwargs.items():
            if key == 'verbose':
                self.verbose = kwargs[key]
            if key == 'leaf_size':
                self.leaf_size = kwargs[key]

        if len(kwargs) >= 1:
            super().__init__(random_tree=False, **kwargs)
        else:
            super().__init__(leaf_size=self.leaf_size, verbose=self.verbose, random_tree=False)

    def author(self):
        return "jadams334"
