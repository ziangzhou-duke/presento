import argparse
import numpy as np
import sys
import wave
from deepspeech import Model


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # loop through each character
    for i, token in enumerate(metadata.tokens):
        # append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # <SIL>
                word_dict = dict()
                word_dict["word"] = "<SIL>"
                word_dict["start_time"] = round(word_start_time, 4)
                word_dict["duration"] = round(token.start_time - word_start_time, 4)
                word_list.append(word_dict)

                # log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            word_dict = dict()
            word_dict["word"] = word
            word_dict["start_time"] = round(word_start_time, 4)
            word_dict["duration"] = round(word_duration, 4)
            word_list.append(word_dict)

            # reset
            word = ""
            word_start_time = token.start_time  # remember the start time of current silence

    return word_list


def postprocess_metadata(metadata):
    res = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return res


def stt(model_path, audio, beam_width=None, scorer_path=None, lm_alpha=None, lm_beta=None, hot_words=None):
    ds = Model(model_path)

    if beam_width:
        ds.setBeamWidth(beam_width)

    desired_sample_rate = ds.sampleRate()

    if scorer_path:
        ds.enableExternalScorer(scorer_path)

        if lm_alpha and lm_beta:
            ds.setScorerAlphaBeta(lm_alpha, lm_beta)

    if hot_words:
        print('Adding hot-words', file=sys.stderr)
        for word_boost in hot_words.split(','):
            word, boost = word_boost.split(':')
            ds.addHotWord(word, float(boost))

    fin = wave.open(audio, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(f'ERROR: original sample rate ({fs_orig}) is different than {desired_sample_rate}hz.', file=sys.stderr)
        exit(1)

    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()

    print('Running inference.', file=sys.stderr)
    res = ds.sttWithMetadata(audio, 3)
    res = postprocess_metadata(res)
    return res


def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False,
                        help='Path to the external scorer file')
    parser.add_argument('--audio', required=True,
                        help='Path to the tests file to run (WAV format)')
    parser.add_argument('--beam_width', type=int,
                        help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--hot_words', type=str, help='Hot-words and their boosts.')
    args = parser.parse_args()

    res = stt(args.model, args.audio, args.beam_width, args.scorer, args.lm_alpha, args.lm_beta, args.hot_words)
    return res


if __name__ == '__main__':
    main()
