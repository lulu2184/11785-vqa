import errno
import json
import os
import sys
import pickle
import re


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"}
manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']
cache_dir = 'data/cache'


def process_punctuation(text):
    output = text
    for p in punct:
        if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip,
                                                              text) is not None):
            output = text.replace(p, '')
        else:
            output = text.replace(p, ' ')
    output = period_strip.sub('', output, re.UNICODE)
    return output


def process_digit_article(text):
    outputs = []
    split_text = text.lower().split()
    for word in split_text:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outputs.append(word)
        else:
            pass
    for idx, word in enumerate(outputs):
        if word in contractions:
            outputs[idx] = contractions[word]
    output = ''.join(outputs)
    return output


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dataset, min_occurrance=9):
    occurrance = {}

    for answer_entry in answers_dataset:
        # answers = answer_entry['answers']
        groud_truth = preprocess_answer(answer_entry['multiple_choice_answer'])
        if groud_truth not in occurrance:
            occurrance[groud_truth] = set()
        occurrance[groud_truth].add(answer_entry['question_id'])

    for answer in list(occurrance):
        if len(occurrance[answer]) < min_occurrance:
            occurrance.pop(answer)

    return occurrance


def create_answer_to_label(occurrance, name):
    answer_to_label = {}
    label_to_answer = []
    label = 0
    for answer in occurrance:
        label_to_answer.append(answer)
        answer_to_label[answer] = label
        label += 1

    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    cache_file = os.path.join(cache_dir, name + '_answer_to_label.pkl')
    pickle.dump(answer_to_label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_dir, name + '_label_to_answer.pkl')
    pickle.dump(label_to_answer, open(cache_file, 'wb'))
    return answer_to_label


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def compute_target(answers_dataset, answer_to_label, name):
    target = []
    for answer_entry in answers_dataset:
        answers = answer_entry['answers']
        answer_count = {}
        for ans in answers:
            answer = ans['answer']
            answer_count[answer] = answer_count.get(answer, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in answer_to_label:
                continue
            labels.append(answer_to_label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append({
            'question_id': answer_entry['question_id'],
            'image_id': answer_entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise
    cache_file = os.path.join(cache_dir, name + '_target.pkl')
    pickle.dump(target, open(cache_file, 'wb'))


if __name__ == '__main__':
    train_answer_file = 'data/v2_mscoco_train2014_annotations.json'
    train_answers = json.load(open(train_answer_file))['annotations']

    val_answer_file = 'data/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']

    train_question_file = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
    train_questions = json.load(open(train_question_file))['questions']

    val_question_file = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
    val_questions = json.load(open(val_question_file))['questions']

    answers = train_answers + val_answers
    occurrance = filter_answers(answers)
    answer_to_label = create_answer_to_label(occurrance, 'trainval')
    compute_target(train_answers, answer_to_label, 'train')
    compute_target(val_answers, answer_to_label, 'val')
