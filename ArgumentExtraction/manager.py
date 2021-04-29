from src.transformers_ML.config import *
from src.transformers_ML.utils import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.model_handler import *
from src.data_parser.utils import ParseEssays, ParseWebDiscourse

from sklearn.linear_model import LogisticRegression

# svm dependencies
from src.classical_ML.model_handler import * #train_svm, svm_train_and_predict
from src.classical_ML.src.data_reader import ProcessRowSentences

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# CUDA_VISIBLE_DEVICES=1
def argNonArgMap(y):
    tmp = []
    for c in y:
        if c == 'c':
            tmp.append(1)
        elif c == 'p':
            tmp.append(1)
        else:
            tmp.append(0)
    return tmp

def GetTrainTestLists(x, y, fold=5):
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []

    test_data_length = len(x) // fold
    for index in range(fold):

        if (index + 1) * test_data_length <= len(x):
            x_test = x[index * test_data_length : (index + 1) * test_data_length]
            y_test = y[index * test_data_length : (index + 1) * test_data_length]
            x_train = x[ : index * test_data_length] + x[(index + 1) * test_data_length : ]
            y_train = y[ : index * test_data_length] + y[(index + 1) * test_data_length : ]
        else:
            x_test = x[index * test_data_length : ]
            y_test = y[index * test_data_length : ]
            x_train = x[ : index * test_data_length]
            y_train = y[ : index * test_data_length]

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        print(len(x_train), len(x_test), len(y_train), len(y_test))

    return x_train_list, x_test_list, y_train_list, y_test_list


def StackSvmBert():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
        essays_sentences = json.load(f)
        y_essays = argNonArgMap([sent['sent-class'] for sent in essays_sentences])

    with open(data_dir + 'web_discourse.json', encoding='utf-8') as f:
        web_d_sentences = json.load(f)
        y_web_d = argNonArgMap([sent['sent-class'] for sent in web_d_sentences])
        web_d_sentences = ProcessRowSentences([sent['sent-text'] for sent in web_d_sentences])

    sentences_all = essays_sentences # web_d_sentences
    y = y_essays  # y_web_d  

    temp_text, test_text, temp_labels, test_labels = train_test_split(sentences_all, y,
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = 0.25,
                                                                            stratify = y)

    '''
    train_text, val_text, train_labels, val_labels = train_test_split(temp_text, temp_labels,
                                                                            random_state = 2018,
                                                                            test_size = 0.20,
                                                                            stratify = temp_labels)
    meta_model = LogisticRegression() # svm.SVC(kernel='linear')
    bert_model, y_bert = bert_train_and_predict(train_text, train_labels, val_text)
    svm_model, y_svm = svm_train_and_predict(train_text, train_labels, val_text)
    '''
    print('training bert and svm ...')
    temp_text, temp_labels = OversamplingGenerator(temp_text, temp_labels)
    # temp_text, temp_labels = np.array(temp_text), np.array(temp_labels)
    bert_model, bert_preds = bert_train_and_predict(temp_text, temp_labels, test_text)
    svm_model, svm_preds = svm_train_and_predict(temp_text, temp_labels, test_text)

    meta_fold = 5
    x_train_list, x_test_list, y_train_list, y_test_list = GetTrainTestLists(temp_text, temp_labels, meta_fold)

    x_meta_train, y_meta_train = [], []
    for index in range(meta_fold):
        print(f'meta model data processing: fold {index + 1}')

        train_text, val_text, train_labels, val_labels = x_train_list[index], x_test_list[index], y_train_list[index], y_test_list[index]
        print(len(train_text), len(val_text), len(train_labels), len(val_labels))

        train_text, train_labels = OversamplingGenerator(train_text, train_labels)

        bert_model, y_bert = bert_train_and_predict(train_text, train_labels, val_text)
        svm_model, y_svm = svm_train_and_predict(train_text, train_labels, val_text)
        x_meta_train += [y_svm[idx][:1] + y_bert[idx] for idx in range(len(val_text))]
        y_meta_train += val_labels
        print(len(x_meta_train), len(y_meta_train))


    print('finish meta train data preparation')

    meta_model = LogisticRegression()
    meta_model.fit(x_meta_train, y_meta_train)

    '''
    with open(ARG_EXTRACTION_ROOT_DIR + '/models/stack_meta_model_5_folds.plk', 'wb') as f:
        pickle.dump(meta_model, f)

    model_path = '{}/models/stack_distilbert_weighted.pt'.format(ARG_EXTRACTION_ROOT_DIR)
    torch.save(bert_model.state_dict(), model_path)

    with open(ARG_EXTRACTION_ROOT_DIR + '/models/stack_svm_model.plk', 'wb') as f:
        pickle.dump(svm_model, f)
    '''

    # testing the performance of the whole pipline

    # svm_preds = svm_predict(svm_model, test_text, proba=True)
    # bert_preds = Predict(bert_model, [sent['sent-text'] for sent in test_text], logits_enable=True)

    x_meta_test = [svm_preds[idx][:1] + bert_preds[idx] for idx in range(len(test_text))]
    meta_preds = meta_model.predict(x_meta_test)

    print(classification_report(test_labels, meta_preds, digits=4))
    print(classification_report(test_labels, [int(pred[0] < pred[1]) for pred in svm_preds], digits=4))
    print(classification_report(test_labels, [int(pred[0] < pred[1]) for pred in bert_preds], digits=4))

    print('#'*50)

    svm_preds_labels = [int(pred[0] < pred[1]) for pred in svm_preds]
    bert_preds_labels = [int(pred[0] < pred[1]) for pred in bert_preds]

    print(confusion_matrix(test_labels, meta_preds))  
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in svm_preds]))
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in bert_preds]))

    '''
    print('bert true')
    print('#'*50)
    for index, c in enumerate(test_labels):
        if c == 1 \
        and svm_preds_labels[index] == 0 \
        and bert_preds_labels[index] == 1 :
            print(test_text[index]['sent-text'])
            print('-'*50)

    print('svm true')
    print('#'*50)
    
    for index, c in enumerate(test_labels):
        if c == 1 \
        and svm_preds_labels[index] == 1 \
        and bert_preds_labels[index] == 0 :
            print(test_text[index]['sent-text'])
            print('-'*50)
    '''


def main(args):
    action, obj = args
    if 'train' == action:
        if 'bert' == obj:
            print('Running BERT training...')
            train_model(epochs=3)
        elif 'svm' == obj:
            print('Running SVM training...')
            train_svm()
        elif 'pack' == obj:
            StackSvmBert()

    elif 'test' == action:
        if 'bert' == obj:
            print('Running BERT testing...')
            # test_model_on_news()
            test_model_on_ibm()
        elif 'svm' == obj:
            print('Running SVM testing...')
            pass

    elif 'parse' == action:
        if 'essays' == obj:
            print('Parsing Essays...')
            ParseEssays()

        elif 'webd' == obj:
            print('Parsing Web Discourse...')
            ParseWebDiscourse()
    else:
        print('wrong params has been given!')

if __name__ == '__main__':
    main(sys.argv[1:])