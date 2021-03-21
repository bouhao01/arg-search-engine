from utils.config import *

batch_size = 32
max_seq_len = 128

def LoadBertModel(model_path):
	# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels = 2,
                                                            output_attentions = False, output_hidden_states = False)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    return model

#########################################################
# class to transform row texts into dataloader
#########################################################
class RowSentencesHandler():
    def __init__(self):
        # bert tokenizer
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        # roberta tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base', do_lower_case=True)

    def GetDataLoader(self, sentences):
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

        # wrap tensors
        data = TensorDataset(sequence, mask)
        # sampler for sampling the data during training
        sampler = SequentialSampler(data)
        # dataLoader for validation set
        dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
        return dataloader

#########################################################
# Function to use the model for prediction
#########################################################
def Predict(model, sentences):
    model.eval()

    text_handler = RowSentencesHandler()
    dataloader = text_handler.GetDataLoader(sentences)
    prediction_result = np.array([])

    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        prediction_result = np.append(prediction_result, np.argmax(logits, axis=1).flatten())

    return prediction_result.tolist()

def TextToSentences(text):
    document = nlp(text)
    sentences = [s.text for s in document.sents if len(s) > 20]
    return sentences