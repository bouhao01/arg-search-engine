from src.transformers_ML.config import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.utils import *

import en_core_web_sm
nlp = en_core_web_sm.load()

OUTPUT_DIR = './output'

def mapping(y):
	conversion = ['n', 'c', 'p']
	ans = [conversion[int(p)] for p in y]
	return ans

sentences = [
		'We should protect endangered species.'
		, 'Centaurea pinnata is an endangered species of plant present in this mountain range.'
		,'We should adopt vegetarianism.'
		,'There are several categories for which the food entry can be classified Appetizers'
		,'Soups and Salads, Seafood, Entrees, Vegetarian Entree, Desserts, and the coined Best Damned Dish.'
		]

def init():
	global model
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 3,
                                                        output_attentions = False, output_hidden_states = False)
	model = model.to(device)
	model.load_state_dict(torch.load('./models/c-p-n_trained_model_epoch_2.pt'))
	print(Predict(model, sentences))
	import time 
	time.sleep(10000)

def main():
	if not os.path.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	inputFile = open('./articles_20201015_20201026.json', 'r', encoding='utf8')

	try:
		lines = inputFile.readlines()
		for index, line in enumerate(lines[:100]):
			dataObject = json.loads(line)
			try:
				fullText = dataObject['full-text']
			except:
				print('document corrupted!!')
				fullText = ''

			# text processing is here
			# document = nlp(fullText)
			sentences_str = [sent.strip() for sent in fullText.split('\r\n') if len(sent) > 15]
			
			tot_sentences = []
			for sent in sentences_str:
				document = nlp(sent)
				tot_sentences += [s.text for s in document.sents if len(s) > 15]
			
			preds = Predict(model, tot_sentences)
			preds = mapping(preds)

			with open(OUTPUT_DIR + f'/doc_{index}_ann.txt', 'w', encoding='utf8') as f:
				for index, sent in enumerate(tot_sentences):
					f.write('{}\t{}\n'.format(sent, preds[index]))

	except Exception as e:
		print('Error: {}'.format(e))
	finally:
		print('Finish')
		inputFile.close()

if __name__ == '__main__':
	init()
	main()