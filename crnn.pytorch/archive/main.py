import csv
import re
from glob import glob

import docx
from nltk.corpus import words, wordnet


filter_words = ["with","been","been","very","with","with","with","left","about","weak","very","with","which","with","were","upon","with","from","been","with","from","been","well","much","side","been","left","above","about","last","some","with","above"]

def add_to_dict_from_line(line, year, word_dictionary):

    words_in_line = []

    for word in line.split():
        word = word.rstrip(".")
        word = word.rstrip(",")
        word = word.rstrip(";")
        word = word.rstrip(":")
        word = word.rstrip("\"")
        word = word.lstrip("\"")
        word = word.rstrip(")")
        word = word.lstrip("(")
        word = word.rstrip("-")
        word = word.rstrip("_")

        if len(word) < 4:
            continue

        if word in filter_words:
            continue

        word = word.lower()
        if word in words.words() or wordnet.synsets(word):
            # if key not in word_dictionary:
            #     word_dictionary[key] = 0

            # word_dictionary[key] += 1
            # return
            words_in_line.append(word)

    words_in_line = sorted(words_in_line)

    for i in range(len(words_in_line)):
        for j in range(i+1,len(words_in_line)):
            key = words_in_line[i] + "_" + words_in_line[j] + "_" + str(year)
            if key not in word_dictionary:
                word_dictionary[key] = 0
            word_dictionary[key] += 1


    # print("word : " + word)


def add_to_dict(word, year, word_dictionary):
    word = word.rstrip(".")
    word = word.rstrip(",")
    word = word.rstrip(";")
    word = word.rstrip(":")
    word = word.rstrip("\"")
    word = word.lstrip("\"")
    word = word.rstrip(")")
    word = word.lstrip("(")
    word = word.rstrip("-")
    word = word.rstrip("_")

    word = word.lower()

    key = word + "_" + str(year)

    if len(word) < 4:
        return

    if word in words.words() or wordnet.synsets(word):
        if key not in word_dictionary:
            word_dictionary[key] = 0

        word_dictionary[key] += 1
        return

    print("word : " + word)


def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "".join(fullText)


word_dictionary = {}

file_names = glob("/Users/rohuntripathi/Downloads/Transciptions/*/*/*.docx", recursive=True)
file_names += glob("/Users/rohuntripathi/Downloads/Transciptions/*/*/*/*.docx", recursive=True)

source = "([0-9]{4})"

print("Total number of files : " + str(len(file_names)))

for index, file_name in enumerate(file_names):

    if not "docx" in file_name:
        continue

    year = re.findall(source, file_name)

    if year is []:
        pass

    year = year[0]

    print("On the following file number : " + str(index))

    line = getText(file_name)
    if line == "":
        continue

    add_to_dict_from_line(line, year, word_dictionary)


rows = [["Word", "word_Two", "Year", "Frequency"]]
for key in word_dictionary:
    terms = key.split("_")
    rows.append([terms[0], terms[1], terms[2], word_dictionary[key]])

with open("csv_out.csv", "w") as fp:
    writer = csv.writer(fp)
    writer.writerows(rows)
