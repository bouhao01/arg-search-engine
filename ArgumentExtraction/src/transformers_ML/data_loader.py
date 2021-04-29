from src.transformers_ML.config import *

data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'
batch_size = 32
max_seq_len = 128


def OversamplingGenerator(x_train, y_traing):
    try:
        x_train = x_train.tolist()
        y_traing = y_traing.tolist()
    except:
        pass

    count_0 = y_traing.count(0)
    count_1 = y_traing.count(1)

    if count_0 < count_1:
        target_count = count_1
        minor_class = 0
    else:
        target_count = count_0
        minor_class = 1

    x_over, y_over = x_train, y_traing

    index = 0
    while True:
        # print(y_over.count(minor_class))
        if y_over.count(minor_class) >= target_count:
            break
        if minor_class == y_traing[index]:
            x_over.append(x_train[index])
            y_over.append(y_traing[index])

        if index < len(x_train):
            index += 1 
        else:
            index = 0

    indices = list(range(target_count * 2))
    random.shuffle(indices)
    x_final = [x_over[i] for i in indices]
    y_final = [y_over[i] for i in indices]
    return x_final, y_final

class DataLoadHandler():

    def __init__(self, test_size = 0.25):

        ess_df = pd.read_json(data_dir + 'essays.json')
        ess_df = ess_df[['sent-text', 'sent-class']]
        web_df = pd.read_json(data_dir + 'web_discourse.json')
        sentences_df = pd.concat([ess_df, web_df]) # ess_df

        # sentences_df = pd.read_json(data_path)
        sentences_df['sent-class'] = sentences_df['sent-class'].map({'n':0, 'c':1, 'p':1})
        X, y = sentences_df['sent-text'], sentences_df['sent-class']


        train_text, test_text, train_labels, test_labels = train_test_split(X, y,
                                                                            random_state = 2018,
                                                                            shuffle = True,
                                                                            test_size = test_size,
                                                                            stratify = y)

        


        # x_train_over , y_train_over = OversamplingGenerator(train_text, train_labels)
        # self.train_text, self.train_labels = np.array(x_train_over), np.array(y_train_over)
        self.train_text, self.train_labels = x_train_over, y_train_over
        self.test_text, self.test_labels = test_text, test_labels
        self.TokenizeAndEncode()

    def TokenizeAndEncode(self):
        if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
                tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        # tokenize and encode sequences in the training se
        self.tokens_train = tokenizer.batch_encode_plus(
            self.train_text.tolist(),
            max_length = max_seq_len,
            add_special_tokens = True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        # tokenize and encode sequences in the validation set
        self.tokens_test = tokenizer.batch_encode_plus(
            self.test_text.tolist(),
            max_length = max_seq_len,
            add_special_tokens = True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

    def GetDataLoaders(self, batch_size = 32):
        # for train set
        train_seq = torch.tensor(self.tokens_train['input_ids'])
        train_mask = torch.tensor(self.tokens_train['attention_mask'])
        train_y = torch.tensor(self.train_labels.tolist())
        # for test set
        test_seq = torch.tensor(self.tokens_test['input_ids'])
        test_mask = torch.tensor(self.tokens_test['attention_mask'])
        test_y = torch.tensor(self.test_labels.tolist())

        # wrap tensors
        train_data = TensorDataset(train_seq, train_mask, train_y)
        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)
        # dataLoader for train set
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)
        
        # wrap tensors
        test_data = TensorDataset(test_seq, test_mask, test_y)
        # sampler for sampling the data during training
        test_sampler = SequentialSampler(test_data)
        # dataLoader for validation set
        test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)

        return (train_dataloader, test_dataloader)

    def GetClassWeights(self):
        #compute the class weights
        class_wts = compute_class_weight('balanced', np.unique(self.train_labels), self.train_labels)
        # convert class weights to tensor
        weights = torch.tensor(class_wts,dtype=torch.float)
        weights = weights.to(device)
        return weights

class RowSentencesHandler():
    def __init__(self):
        if 'bert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilbert-base-uncased' == TRANSFORMERS_MODEL_NAME:
            tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'roberta-base' == TRANSFORMERS_MODEL_NAME:
                tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        elif 'distilroberta-base' == TRANSFORMERS_MODEL_NAME:
            tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMERS_MODEL_NAME, do_lower_case=True)

        self.tokenizer = tokenizer

    def GetDataLoader(self, sentences, labels=None):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            max_length = max_seq_len,
            add_special_tokens = True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        sequence = torch.tensor(tokens['input_ids']) #.to(device)
        mask = torch.tensor(tokens['attention_mask']) #.to(device)
        if labels:
            y_labels = torch.tensor(labels)

        # wrap tensors
        if labels:
            data = TensorDataset(sequence, mask, y_labels)
        else:
            data = TensorDataset(sequence, mask)

        # sampler for sampling the data during training
        sampler = SequentialSampler(data)
        # dataLoader for validation set
        dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
        return dataloader
