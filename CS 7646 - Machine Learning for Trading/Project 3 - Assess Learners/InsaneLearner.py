# Import allowed per Steven B. Bryant https://piazza.com/class/kdthusf8jeo7ia?cid=109_f2
import BagLearner
class InsaneLearner(BagLearner.BagLearner):
    def __init__(self, verbose=False, **kwargs):
        self.insane = True
        self.verbose = verbose
        super().__init__(verbose, insane=True, **kwargs)
    def author(self):
        return 'jadams334'
