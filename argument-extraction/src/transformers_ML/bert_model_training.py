from src.config import *
# from src.bert_model import BERT_Arch, train, evaluate
from src.data_loader import DataLoadHandler
from src.utils import *

data_loader = DataLoadHandler(data_path='combined_sentence.json')
train_dataloader, test_dataloader = data_loader.GetDataLoaders()
class_weights = data_loader.GetClassWeights()



config = BertConfig.from_json_file('bert_config.json')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
# print(model.parameters)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2,
                                                        output_attentions = False, output_hidden_states = False)
# Saving model config
# config = model.config
# config.to_json_file('bert_config.json')




model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

for epoch_i in range(epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    train_loss = TrainBertSeqCl(model, train_dataloader, optimizer, scheduler, class_weights)
    loss_values.append(train_loss)
    
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print('validation with training data: ')
    EvaluateBertSeqCl(model, train_dataloader)

    # ========================================
    #             Model Saving
    # ========================================
    torch.save(model.state_dict(), f'./models/essay-a-n_trained_model_epoch_{epoch_i}_weighted.pt')


print("Training complete!")

print('validation with testing data: ')
EvaluateBertSeqCl(model, test_dataloader)




