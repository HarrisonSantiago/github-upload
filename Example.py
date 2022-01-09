from Augmenter import *
from Train import *
from get_classifiers import *
from ProcessInput import *

training_filepath = ["CORAAL_cleaned_txt_files_be_sorted_habitual1_0.csv"]

training_set = processer(training_filepath, habcol = 'Habituality')

classifiers = get_classifiers(training_set)

test_filepath = ["test.csv"]

test_set = processer(test_filepath)

predictions = classifiers['Ensemble'].predict(test_set)
