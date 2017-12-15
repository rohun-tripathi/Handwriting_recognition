import refactored_main, operator

from spell import correction

import numpy as np

from glob import glob

from collections import defaultdict
import functools
# Get Opt
opt = refactored_main.get_parameters()

# Init CRNN
crnn, converter, _ = refactored_main.load_trained_crnn_for_eval(opt)

# Image mapping for proof of concept
image_mapping = {'1': './data/words/p01/p01-174/*.png', '2': './data/words/p02/p02-000/*.png',
                 '3': './data/words/p06/p06-047/*.png'}

# curate documents
doc_to_text = defaultdict(int)
for file in glob('data/med_ground_truth/*.txt'):
    wordcount = defaultdict(int)
    for word in open(file).read().lower().split():
        wordcount[word] += 1

    doc_to_text[file] = wordcount


def get_most_relevant(keyword_string, top_needed=3):
    keys = keyword_string.split(",")

    score_list = []
    for file in doc_to_text:
        file_score = functools.reduce(operator.mul, [doc_to_text[file][keyword.strip()] for keyword in keys], 1)
        if file_score > 0:
            score_list.append((file_score, file))

    if len(score_list) == 0:
        return None, None

    most_frequent = sorted(score_list, key=lambda x: x[0], reverse=True)
    return [{'name': most_frequent[k][1].split("/")[-1].rstrip(" copy.txt"), 'text': open(most_frequent[k][1]).read()}
            for k in range(min(top_needed, len(most_frequent)))]


def check_creds(username, password):
    # Check login creds
    # For now, Proof of concept only
    return username == "admin" and password == "admin"


def extract_for_image(extra_path):
    predicted_list = refactored_main.extract_result(opt, crnn, converter, extra_path)

    match = np.array(
        [1 if prediction.target.lower() == prediction.pred.lower() else 0 for prediction in predicted_list])
    print("Model Prediction : ")
    model_prediction = " ".join([prediction.pred.lower() for prediction in predicted_list])
    print(model_prediction)
    print('Accuracy : ' + str(match.mean()))

    corrected_text = [correction(prediction.pred.lower()) for prediction in predicted_list]

    corrected_match = np.array([1 if corrected.lower() == prediction.target.lower() else 0 for corrected, prediction in
                                zip(corrected_text, predicted_list)])
    print("Corrected Prediction : ")
    corrected_prediction = " ".join(corrected_text)
    print(corrected_prediction)
    print('Accuracy : ' + str(corrected_match.mean()))

    return "\n\n".join(["Original: " + model_prediction, corrected_prediction])


def extract_result(image_index):
    predicted_list = refactored_main.extract_result(opt, crnn, converter, image_mapping[image_index])

    match = np.array([1 if prediction.target.lower() == prediction.pred.lower() else 0 for prediction in predicted_list])
    print("Model Prediction : ")
    print(" ".join([prediction.pred.lower() for prediction in predicted_list]))
    print('Accuracy : ' + str(match.mean()))

    corrected_text = [correction(prediction.pred.lower()) for prediction in predicted_list]

    corrected_match = np.array([1 if corrected.lower() == prediction.target.lower() else 0 for corrected, prediction in zip(corrected_text, predicted_list)])
    print("Corrected Prediction : ")
    print(" ".join(corrected_text))
    print('Accuracy : ' + str(corrected_match.mean()))

    return " ".join(corrected_text)


if __name__ == '__main__':
    # print(get_most_relevant('disease, wound, spinal'))
    print(extract_result('1'))
