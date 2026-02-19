# --- Cell 1 ---
import pandas as pd #Pandas is for importing the .csv file
import torch
from torch.utils.data import Dataset, DataLoader #This will be used to preprocess the .csv file. This is just a convenience way to use them
import numpy as np #We will be using NumPy for many function

# Use GPU if available or the MacOS MPS equivalent
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using {} device".format(device))

# --- Cell 2 ---
## Code for Part 1
## Any imports you need can go here
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

class CustomTextDataset(Dataset):
    def __init__(self, csv_file):
    ##Q1
        df = pd.read_csv(csv_file)
       #uncomment for more output data print(df)

        self.description = df["Description"].tolist()    ## link the desciption column to a variable called description etc
        self.urgency = df["Urgency"].tolist()
        self.cost = df["Cost"].tolist()
        self.resolution = df["Resolution"].tolist()
        self.categorys = df["Category"]

       #uncomment for more output data print("\n\nDescription:\n", self.description)
       #uncomment for more output data print("\nUrgency:\n", self.urgency)
       #uncomment for more output data print("\nCost:\n", self.cost )
       #uncomment for more output data print("\nResolution:\n", self.resolution)
       #uncomment for more output data print("\nCategory:\n", self.categorys )
         
        ## Convert categorical variables into integer values and print out categories
        ##Q3
        label_encoder = preprocessing.LabelEncoder()
        
        self.urgency_integer = label_encoder.fit_transform(self.urgency)   ###### encode the columns into numeric values of 0,1,2,3,4 etc
        self.category_integer = label_encoder.fit_transform(self.categorys)
        self.resolution_integer = label_encoder.fit_transform(self.resolution)
        
        self.category_integer = torch.as_tensor(self.category_integer, dtype=torch.long)
  
        ##print("\nUrgencys Intergers\n", self.urgency_integer)
        ##print("\nCategorys Intergers\n", self.category_integer)
        ##print("\nResolutions Intergers\n", self.resolution_integer)
        
        ## Preprocess and tokenize text
        tokenizer = RegexpTokenizer(r'\w+')   ## setting up the tokenizer we will use the regexp version
        stop_words = set(stopwords.words('english'))  ## use a preset to get rid of stop words
        
        self.tokenized_description = []

        for description in self.description:
            tokens = tokenizer.tokenize(description.casefold()) ## set all data to lowercase
            self.tokenized_description.append(tokens)  
            ##print(tokenized_description) for testing

       
           
        ## Write your code for Q5 here
        stop_words = set(stopwords.words('english'))
        tokenized_description_nostopwords = []   ## setting the max length of each sentence in the description column
        maxlength = 0
        for descript in self.tokenized_description:
            if len(descript) > maxlength:
                maxlength = len(descript)
             ##   print( maxlength)  for testing    
            
        for description in self.tokenized_description:
            temp_sentence = []
            for token in description:
                if token not in stop_words:
                    temp_sentence.append(token)
            tokenized_description_nostopwords.append(temp_sentence)
        
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("scarves")
        lemm_words = []
        for word in description:
            lemm_word = lemmatizer.lemmatize(word,pos='v')
            #print(lemm_word)
            lemm_words.append(lemm_word)

          ##  maxlength = max([len(description) for description in tokenized_description_nostopwords])
          ##  print(maxlength)
          ##  print(tokenized_description_nostopwords)  -- TESTING STAGES OF TOKENIZATION
        
         ## Add EOS and PAD tokens Q6
        tokenized_description_nostopwords_samelength = []
        for description in tokenized_description_nostopwords:
            temporary_copy = description.copy() ##makes each description have the same number of tokens
            temporary_copy.append('<eos>') ## makes a copy of description class class to allow for seperate testing
                        
            while len(temporary_copy) < maxlength+1:         ## adding a PAD to the sentences that arent max length       
                temporary_copy.append('<pad>')
                
            #print(temporary_copy)
            tokenized_description_nostopwords_samelength.append(temporary_copy)

        self.new_description = tokenized_description_nostopwords_samelength  ## create new variable to allow previous testing to work
                   
        ## One-hot encode text
        ## Write your code for Q7 here
        text_array = []
        max_length = 0
        
        for current_item in self.description:
            tokenizedsentence = tokenizer.tokenize(current_item.casefold())
            tokenizedsentence.append('<eos>')  ## using casefold again as well as adding end of sequence to every sentence
            
            if len(tokenizedsentence) > max_length:
                max_length = len(tokenizedsentence)

            text_array.append(tokenizedsentence)
            
        text_array2 = []         ## adds a pad and oes to match out the sentence length of each description
        for item in text_array:
            while len(item) < max_length:
                item.append('<pad>')
            text_array2.append(item)  ## addsd multiple pads aslong as sentence is smaller than max length
            
        #print(text_array2) for testing 

        ## Create a dictionary for all words
        ## This is step 2 of Q7
        self.unique_tokens = sorted(set([token for description in text_array2 for token in description]))
        for sentence in ['<eos>', '<pad>']:
            if sentence not in self.unique_tokens: ## groups all pads and eos as 1 encode
                self.unique_tokens.append(sentence)
                
        word_dict = {token: idx for idx, token in enumerate(self.unique_tokens)}
        self.word_dict = word_dict  ## create a word dictionary

        len_data = len(self.new_description)
        max_sentence_len = len(self.new_description[0])
        embed_dim = len(self.unique_tokens)

        descriptionoh = np.zeros((
            len_data, max_sentence_len, embed_dim), 
            dtype=np.float32
        ) 
        for i, description in enumerate(self.new_description):
            for j, token in enumerate(description):
                idx = word_dict[token]
                descriptionoh[i, j, idx] = 1.0         
        ###############################
        ## Write your code for Q9 (optional) here
        ###############################
        ## Convert to torch tensors
        self.descriptionoh = torch.as_tensor(descriptionoh)
   ## print(self.descriptionoh.shape)  
   ## print(self.descriptionoh[0])
            
    def __len__(self):
        return len(self.category_integer)
    def __getitem__(self, idx):
                description = self.descriptionoh[idx]
                target = self.category_integer[idx]
                sample = [description, target]
                return sample
                Dataset = CustomTextDataset("factoryReports_export.csv")

# --- Cell 3 ---
# Create the object to test that it all works
all_data_object = CustomTextDataset("factoryReports_export.csv")

# Display image and label.
print('\nFirst iteration of data set: ', next(iter(all_data_object)), '\n')
print('Dataset size = {}'.format(str(len(all_data_object))))

# --- Cell 4 ---
from torch import nn
import math
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, embedDim):
        super().__init__()
        # Write your code for Q1 here
        
        self.embedDim = embedDim
        self.query = nn.Linear(embedDim, embedDim, bias=False)  ## using the linear function  in the torch package
        self.key = nn.Linear(embedDim, embedDim, bias=False)   ## the bias has to be false
        self.value = nn.Linear(embedDim, embedDim, bias=False)
    
    def forward(self, x):
        # Write your code for Q2 here
        
        Q = self.query(x)  ##new variables ot represent the dimensions of the attention
        K = self.key(x)
        V = self.value(x)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.embedDim)  ## the equation to find the logits

        att = F.softmax(attn_logits, dim=-1) ## softmax to represent the data smoothly
        h = torch.matmul(att, V)
        return h, att
        
embedDim = 10   ## preset values for testing in the criteria
seq_len = 5
x = torch.randn(seq_len, embedDim)

test = SelfAttention(embedDim)
h, att = test(x)

sns.set_theme()
ax = sns.heatmap(att.detach().numpy(), xticklabels=['w1','w2','w3','w4','w5'],yticklabels=['w1','w2','w3','w4','w5'])
#print(h.shape, att.shape) for testing

# --- Cell 5 ---
from torch import nn
import math
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch

class SelfAttention_reduced(torch.nn.Module):
    #Complete the functions below to write your code for Q3
    def __init__(self, inputDim, embedDim):
        super().__init__()
 ## same code as above however there is now an input dimension
        self.inputDim = inputDim          ##new input vairable added for dimensions
        self.embedDim = embedDim
        self.query = nn.Linear(inputDim, embedDim, bias=False)
        self.key = nn.Linear(inputDim, embedDim, bias=False)
        self.value = nn.Linear(inputDim, embedDim, bias=False)
        
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.embedDim)

        att = F.softmax(attn_logits, dim=-1)
        h = torch.matmul(att, V)
        
        return h,att

# --- Cell 6 ---
## Test code for Q3 and Q4
testTensor = torch.tensor([[[0.0062, 0.2186],
         [0.7316, 0.2092],
         [0.8501, 0.7958]],

        [[0.9522, 0.6007],
         [0.3869, 0.9120],
         [0.2445, 0.5242]],

        [[0.8930, 0.3641],
         [0.1240, 0.3233],
         [0.6844, 0.8215]],

        [[0.4050, 0.5804],
         [0.3892, 0.6785],
         [0.4283, 0.6082]]])

#testClass = SelfAttention(testTensor.size(-1))
testClass = SelfAttention_reduced(testTensor.size(-1),5)  #changing the 5 to a 2 creates the correct matrice as shown
h, attn = testClass.forward(testTensor)    
   
print(testTensor.shape)
print(h.shape)
print(attn.shape)

# --- Cell 7 ---
class MultiheadSelfAttention_reduced(torch.nn.Module):
    def __init__(self, inputDim, embedDim, noHeads):
        super().__init__()
        # Write your code for Q6 here
        
        self.inputDim = inputDim
        self.embedDim = embedDim ## defining the variables frmo the init 
        self.noHeads = noHeads
        
        self.query = nn.Linear(inputDim, embedDim, bias=False)  ## use self to maintain the data on the variable outside of the class
        self.key = nn.Linear(inputDim, embedDim, bias=False)
        self.value = nn.Linear(inputDim, embedDim, bias=False)
        
        N = embedDim / noHeads   ## for multi head there needs to be a "slice" for each dimension this slice is the embed dimension divided by the number of heads
      ## EmbedDim is divided by the number of heads as each indvidual attention needs equal proportion of embedding
    def forward(self, x):
        # Write your code for Q6 here
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x) 

        Batch_size, seq_len, embedDim = Q.shape  ##the dimension is made up of the size the embedded and the sequence length
                
        # loops through the heads
        for N in range( self.noHeads):
            begin = N * self.noHeads   ## loops through the dimension slices and adds it to the number of heads
            finish = (N+1) * self.noHeads  ## finishes at 1+ the embedded/ number of heads
            
            # takes a slice from each of the heads and puts them in an array
            Q2 = Q[:, :, begin:finish]
            K2 = K[:, :, begin:finish]
            V2 = V[:, :, begin:finish]
            
            # calculates the attention for the new heads
            attn_logits = torch.matmul(Q2, K2.transpose(-2, -1))
            attn_logits = attn_logits / math.sqrt(self.noHeads)
            att = F.softmax(attn_logits, dim=-1)  ## representation of equations above to utilise multi head
        ## softmax to view the data nicely
        # create empty tensor to fill will head slices
        h = torch.zeros(Batch_size, seq_len, self.embedDim)
        
        # store all the attentions matrices in tensor
        att3 = torch.zeros(Batch_size, self.noHeads, seq_len, seq_len)            

        return h,att3

# --- Cell 8 ---
## Test code for Q6 and Q7
testTensor = torch.tensor([[[0.0062, 0.2186],
         [0.7316, 0.2092],
         [0.8501, 0.7958]],

        [[0.9522, 0.6007],
         [0.3869, 0.9120],
         [0.2445, 0.5242]],

        [[0.8930, 0.3641],
         [0.1240, 0.3233],
         [0.6844, 0.8215]],

        [[0.4050, 0.5804],
         [0.3892, 0.6785],
         [0.4283, 0.6082]]])

#testClass = SelfAttention(testTensor.size(-1))
testClass = MultiheadSelfAttention_reduced(testTensor.size(-1),6,2)
h, att3 = testClass.forward(testTensor)

print(testTensor.shape)
print(h.shape)
print(att3.shape)

# --- Cell 9 ---
##################
###### everything below until the nueral network class is directly taken from question 1 and 2 
##it is the code to process and tokenize the data then use a multi head attention class. this is becuase the cells wouldnt load any of the code in the kernals above
import math
import torch.nn as nn
import torch 
import torch.nn.functional as F
## Code for Part 1
## Any imports you need can go here
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class CustomTextDataset(Dataset):
    def __init__(self, csv_file):
    ##Q1
        df = pd.read_csv(csv_file)
       #uncomment for more output data print(df)

        self.description = df["Description"].tolist()
        self.urgency = df["Urgency"].tolist()
        self.cost = df["Cost"].tolist()
        self.resolution = df["Resolution"].tolist()
        self.categorys = df["Category"]

       #uncomment for more output data print("\n\nDescription:\n", self.description)
       #uncomment for more output data print("\nUrgency:\n", self.urgency)
       #uncomment for more output data print("\nCost:\n", self.cost )
       #uncomment for more output data print("\nResolution:\n", self.resolution)
       #uncomment for more output data print("\nCategory:\n", self.categorys )
         
        ## Convert categorical variables into integer values and print out categories
        ##Q3
        label_encoder = preprocessing.LabelEncoder()
        
        self.urgency_integer = label_encoder.fit_transform(self.urgency)
        self.category_integer = label_encoder.fit_transform(self.categorys)
        self.resolution_integer = label_encoder.fit_transform(self.resolution)
        
        self.category_integer = torch.as_tensor(self.category_integer, dtype=torch.long)
  
        ##print("\nUrgencys Intergers\n", self.urgency_integer)
        ##print("\nCategorys Intergers\n", self.category_integer)
        ##print("\nResolutions Intergers\n", self.resolution_integer)
        

        ## Preprocess and tokenize text
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        
        self.tokenized_description = []

        for description in self.description:
            tokens = tokenizer.tokenize(description.casefold())
            self.tokenized_description.append(tokens)
          ##print(tokenized_description)
           
        ## Write your code for Q5 here
        stop_words = set(stopwords.words('english'))
        
        tokenized_description_nostopwords = []
        maxlength = 0
        for descript in self.tokenized_description:
            if len(descript) > maxlength:
                maxlength = len(descript)
             ##   print( maxlength)      
            
        for description in self.tokenized_description:
            temp_sentence = []
            for token in description:
                if token not in stop_words:
                    temp_sentence.append(token)
            tokenized_description_nostopwords.append(temp_sentence)

          ##  maxlength = max([len(description) for description in tokenized_description_nostopwords])
          ##  print(maxlength)
          ## print(tokenized_description_nostopwords)  -- TESTING STAGES OF TOKENIZATION
          ## Add EOS and PAD tokens Q6
        tokenized_description_nostopwords_samelength = []
        for description in tokenized_description_nostopwords:
            temporary_copy = description.copy() ##makes each description have the same number of tokens
            temporary_copy.append('<eos>')  ## makes copy to allow for testing on both variables old description and new
                        
            while len(temporary_copy) < maxlength+1:               
                temporary_copy.append('<pad>')
                
            #print(temporary_copy)
            tokenized_description_nostopwords_samelength.append(temporary_copy)

        self.new_description = tokenized_description_nostopwords_samelength
                   
        ## One-hot encode text
        ## Write your code for Q7 here
        text_array = []
        max_length = 0
        
        for current_item in self.description:
            tokenizedsentence = tokenizer.tokenize(current_item.casefold())
            tokenizedsentence.append('<eos>')
            
            if len(tokenizedsentence) > max_length:
                max_length = len(tokenizedsentence)

            text_array.append(tokenizedsentence)
            
        text_array2 = []
        for item in text_array:
            while len(item) < max_length:
                item.append('<pad>')
            text_array2.append(item)
            
        #print(text_array2)
        ## Create a dictionary for all words
        
        ## This is step 2 of Q7
        self.unique_tokens = sorted(set([token for description in text_array2 for token in description]))
        for sentence in ['<eos>', '<pad>']:
            if sentence not in self.unique_tokens: ## groups all pads and eos as 1 encode
                self.unique_tokens.append(sentence)
                
        word_dict = {token: idx for idx, token in enumerate(self.unique_tokens)}
        self.word_dict = word_dict

        len_data = len(self.new_description)
        max_sentence_len = len(self.new_description[0])
        embed_dim = len(self.unique_tokens)

        descriptionoh = np.zeros((
            len_data, max_sentence_len, embed_dim), 
            dtype=np.float32
        ) 
        for i, description in enumerate(self.new_description):
            for j, token in enumerate(description):
                idx = word_dict[token]
                descriptionoh[i, j, idx] = 1.0         
        ###############################
        ## Write your code for Q9 (optional) here
        ###############################
        ## Convert to torch tensors
        self.descriptionoh = torch.as_tensor(descriptionoh)
   ## print(self.descriptionoh.shape)  
   ## print(self.descriptionoh[0])
            
    def __len__(self):
        return len(self.category_integer)
    def __getitem__(self, idx):
                description = self.descriptionoh[idx]
                target = self.category_integer[idx]
                sample = [description, target]
                return sample
                Dataset = CustomTextDataset("factoryReports_export.csv") 

class MultiheadSelfAttention_reduced(torch.nn.Module):
    def __init__(self, inputDim, embedDim, noHeads):
        super().__init__()
        # Write your code for Q6 here
        
        self.inputDim = inputDim
        self.embedDim = embedDim
        self.noHeads = noHeads
        
        self.query = nn.Linear(inputDim, embedDim, bias=False)
        self.key = nn.Linear(inputDim, embedDim, bias=False)
        self.value = nn.Linear(inputDim, embedDim, bias=False)


        assert embedDim % noHeads == 0
        self.headDim = embedDim // noHeads
        
        
    
    def forward(self, x):
        
        # Write your code for Q6 here
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x) 
        
        Batch_size, seq_len, _ = Q.shape
        
        # create empty tensor to fill will head slices
        h = torch.zeros(Batch_size, seq_len, self.embedDim)
        
        # store all the attentions matrices
        att3 = torch.zeros(Batch_size, self.noHeads, seq_len, seq_len)
        
        # loops through the heads using lecture instructions
        for N in range(self.noHeads):
            begin = N * self.headDim
            finish = (N+1) * self.headDim
            
            # takes a slice from each of the heads and puts them in an array
            Q2 = Q[:, :, begin:finish]
            K2 = K[:, :, begin:finish]
            V2 = V[:, :, begin:finish]
            
            # calculates the attention for the new heads
            attn_logits = torch.matmul(Q2, K2.transpose(-2, -1))
            attn_logits = attn_logits / math.sqrt(self.headDim)
            att = F.softmax(attn_logits, dim=-1)
            
            # use the attention on V2
            h2 = torch.matmul(att, V2)   # stores the variables to use in the nueral network
            h[:, :, begin:finish] = h2
            att3[:, N, :, :] = att

        return h, att3
        
class NeuralNetwork(torch.nn.Module):
    def __init__(self, textLength, inputDim, embedDim=None, noHeads=1, hiddenDim=0, outputDim=0):
        ##textLength - max length of any input sentence
        ##inputDim   - raw embedding dim (following one-hot encoding) of each word
        ##embedDim   - output embedding dimension following self-attention
        ##noHeads    - number of attention heads (leave out if not using multihead)
        
        super().__init__()
        
        if embedDim is None:
            embedDim = inputDim ## If you are using the self-attention with only one embedding dimension, i.e. the first version you developed
        self.embedDim = embedDim 
        self.textlength = textLength
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim

        self.attn = MultiheadSelfAttention_reduced(inputDim, embedDim, noHeads)  ##calls the variables form, above to be stored
        
        flatDim = textLength * embedDim  ## flattening the layer to pass a matrix as a vector to be represnted
        if hiddenDim > 0: 
            self.w_q = nn.Linear(flatDim, hiddenDim)
            self.w_k = nn.Linear(hiddenDim, outputDim)
        else:
            self.w_q = nn.Linear(flatDim, outputDim)
            self.w_k = None
        

            ##self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        h, attn = self.attn(x)
        hf = h.view(h.size(0), -1)
        if self.hiddenDim > 0:
            z = F.relu(self.w_q(hf))  ## setting the linear layers
            logits = self.w_k(z)
        else:
            logits = self.w_q(hf)
        ##probability = self.softmax(logits)
        return logits, attn

# --- Cell 10 ---
###############################
all_data_object = CustomTextDataset("factoryReports_export.csv")
epochs = 100   #doesnt take to long to produce whilst provided an accurate display of data
batch_size = 60
learning_rate = 0.001  # values have been attained through part 1-3 or through jypnb lecture files as a reference
train_prop = 0.8
train_no = 384   #0.8 * 480 data set = 384
val_no = 96      # 480-384 = 96
###############################

# We will now create training and validation dataloaders
print(f'no train: {train_no}, no val: {val_no}')
train_subset, val_subset = torch.utils.data.random_split(all_data_object, [train_no, val_no], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size)

# --- Cell 11 ---
###############################
## Write your code for Q3 here
###############################
all_data_object = CustomTextDataset("factoryReports_export.csv")
num_samples, maxSentenceLength, inputEmbedding = all_data_object.descriptionoh.shape
nCategories = len(torch.unique(all_data_object.category_integer))
model = NeuralNetwork(
    textLength = maxSentenceLength,
    inputDim = inputEmbedding,
    embedDim = None,
    noHeads = 1,
    hiddenDim=0,
    outputDim = nCategories )
device = torch.device("cpu")
model = model.to(device)

###############################
## Write your code for Q6 here
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
###############################

# --- Cell 12 ---
# The loop that is called for training
def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practice
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        logits, attn = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

# Loop run each epoch to test performance on validation data
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            logits, attn = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Resent parameters so when the model is run it always starts afresh
for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

# Code that actually trains and tests model by calling functions above.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimiser)
    test_loop(val_loader, model, loss_fn)

print("Done!")
