import numpy as np


def filler_words(transcript, fillers=None):
    if fillers is None:
        fillers = ['um', 'ah']

    segments = [tr['word'] for tr in transcript[0]['words']]
    print(segments)
    num_filler = np.sum([segments.count(fl) for fl in fillers])
    total_words = len(segments)
    print('total_words:', total_words)
    print('number of ', fillers, 'said:', num_filler)
    percent = num_filler / total_words
    print('percent of filler words', percent)
    print('compared to TED standard frequency of filler words (0.005589%)...')
    compare_to_standard(percent, 0.005589)  # gold standard is hard coded into the program right now
    return percent


def compare_to_standard(percent, standard):
    if percent < standard:
        print('GOOD')
    else:
        print('BAD')


def analyze(transcript):
    ums = filler_words(transcript)
    # likes = filler_words(preprocessed, 'like') #<-- inaccurate
    silences = filler_words(transcript, ['<SIL>'])
    print('% of filler words)', ums)
    print('% of "<SIL>"', silences)
