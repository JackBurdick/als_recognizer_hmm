import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    xLens = test_set.get_all_Xlengths()

    # Implement the recognizer
    for w_id in range(len(test_set.get_all_sequences())):
        p_dict = {}
        best_score = float('-inf')
        best_w = None
        x, lengths = xLens[w_id]
        for w, m in models.items():
            try:
                p_dict[w] = m.score(x, lengths)
            except:
                p_dict[w] = float('-inf')
            if p_dict[w] >= best_score:
                best_score, best_w = p_dict[w], w
        
        probabilities.append(p_dict)
        guesses.append(best_w)

    return probabilities, guesses
