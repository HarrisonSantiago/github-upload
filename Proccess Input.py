import pandas as pd
import numpy as np
import re

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def input_processor(filePath, textcol = 'Concordance', habcol = None):

    assert type(textcol) == str, 'colname must be a string'
    assert type(textcol) == str or habcol == None, 'colname must be a string'

    toret = []

    for path in filePath:

        if habcol == None:
            text = pd.read_csv(path, usecols=['Concordance'])
        else:
            text = pd.read_csv(path, usecols=[habcol,'Concordance'])

        for index, row in text.iterrows():
            input_row = str(text.iloc[index,1]).lower()
            spaces = findOccurrences(input_row, ' ')

            assert 48 in spaces or 49 in spaces, 'location of be not where expected'

            be_index = spaces.index(48) if 48 in spaces else spaces.index(49)

            if len(spaces) - be_index > 5:

                input_row = " " + input_row + " "
                input_row = re.sub("[^\w\s']", "", input_row) # remove punctuation
                input_row = input_row.replace(" be'", " be ")

                if habcol == None:
                    toret.append([input_row, be_index + 1])
                else:
                    hab = int(text.iloc[index, 0])
                    toret.append([input_row, be_index + 1, hab])



    return np.array(toret)