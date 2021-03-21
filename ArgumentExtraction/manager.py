from src.transformers_ML.config import *
from src.transformers_ML.utils import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.model_handler import *
from src.data_parser.utils import ParseEssays, ParseWebDiscourse

# CUDA_VISIBLE_DEVICES=1

def main(args):
	action, obj = args
	if 'train' == action:
		if 'bert' == obj:
			print('Running BERT training...')
			train_model(epochs=3)
		elif 'svm' == obj:
			print('Running SVM training...')
			pass

	elif 'test' == action:
		if 'bert' == obj:
			print('Running BERT testing...')
			test_model_on_news()
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