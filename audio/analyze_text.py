import numpy as np

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
             'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
             'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
             "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
             'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
             "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
             "wouldn't"]

FILLER_WORDS = ['um', 'umm', 'ummm', 'ah', 'ahh', 'ahhh', 'uh']


def count_words(segments, vals=None):
    if vals is None:
        vals = FILLER_WORDS

    count = 0
    idx = []
    for fl in vals:
        for i, w in enumerate(segments):
            if w == fl:
                idx.append(i)
                count += 1

    total_words = len(segments)
    percent = count / total_words
    return count, percent, idx


def analyze(transcript):
    transcript = transcript[0]['words']
    ret = dict(tot=0, mdsil=0, mdstp=0, siltot=0, unw=0, unwtot=0, nwps=0)

    segments = [tr['word'] for tr in transcript]

    # remove leading and trailing SIL
    n = len(segments)
    i = 0
    j = n - 1
    while i < n and j >= 0:
        if transcript[i]['word'] != '<SIL>' and transcript[j]['word'] != '<SIL>':
            break

        if transcript[i]['word'] == '<SIL>':
            i += 1
        if transcript[j]['word'] == '<SIL>':
            j -= 1
    segments = segments[i:j + 1]
    print(segments)

    # fillers = count_words(segments)

    ret['tot'] = len(segments)
    ret['unw'] = len(list(set(segments)))
    ret['unwtot'] = ret['unw'] / ret['tot']

    tot_dur = 0
    for tr in transcript:
        tot_dur += tr['duration']

    ret['nwps'] = ret['tot'] / tot_dur

    # silence
    n_sil, _, sil_i = count_words(segments, ['<SIL>'])
    sil_durs = [transcript[i]['duration'] for i in sil_i]
    ret['mdsil'] = np.mean(sil_durs)
    ret['siltot'] = np.sum(sil_durs) / tot_dur

    n_stp, stptot, stp_i = count_words(segments, stopwords)
    stp_durs = [transcript[i]['duration'] for i in stp_i]
    ret['mdstp'] = np.mean(stp_durs)
    return ret
