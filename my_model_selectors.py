import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        best_score = float('inf')
        best_model = None
        
        for num_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                cur_model = self.base_model(num_comp)
                logL = cur_model.score(self.X, self.lengths)
                logN = math.log(self.X.shape[0])
                num_params = num_comp * num_comp + 2 * num_comp * self.lengths[0] - 1
                cur_bic_score = -2*logL + num_params * logN
                
                # update best model and score
                if cur_bic_score <= best_score:
                    best_score, best_model = cur_bic_score, cur_model
            except:
                pass
            
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None

        for num_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                cur_model = self.base_model(num_comp)
                logL = cur_model.score(self.X, self.lengths)
                other_scores = []
                for w, XLen in self.hwords.items():
                    if w is not self.this_word:
                        other_w, other_len = self.hwords[w]
                        other_scores.append(cur_model.score(other_w, other_len))

                # DIC score
                if other_scores:
                    cur_dic_score = logL - np.mean(other_scores)
                    if best_score <= cur_dic_score:
                        best_score, best_model = cur_dic_score, cur_model
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None
        cur_split = KFold(n_splits=2)

        # Ensure sequence is long enough to split
        if len(self.sequences) > 1:
            for num_comp in range(self.min_n_components, self.max_n_components+1):
                logLs = []

                for train_idx, test_idx in cur_split.split(self.sequences):
                    train_X, train_len = combine_sequences(train_idx, self.sequences)
                    test_X, test_len = combine_sequences(test_idx, self.sequences)
                    try:
                        cur_cv_model = self.base_model(num_comp)
                        cur_cv_model.fit(train_X, train_len)
                        logLs.append(cur_cv_model.score(test_X, test_len))
                    except:
                        pass
                
                # k_fold score
                if logLs:
                    cur_kf_score = np.mean(logLs)
                    if cur_kf_score >= best_score:
                        best_score, best_model = cur_kf_score, cur_cv_model
        
        return best_model
