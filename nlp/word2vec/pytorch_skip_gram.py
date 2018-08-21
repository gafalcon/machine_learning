import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow ,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now ,
Will be a totter'd weed of small worth held :
Then being asked , where all thy beauty lies ,
Where all the treasure of thy lusty days ;
To say , within thine own deep sunken eyes ,
Were an all-eating shame , and thriftless praise .
How much more praise deserv'd thy beauty's use ,
If thou couldst answer ' This fair child of mine
Shall sum my count , and make my old excuse ,'
Proving his beauty by succession thine !
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".lower().split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
return out

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embedding_w_c = nn.Embedding(vocab_size, embedding_dim) #Embeddings of center word
        self.embedding_u_o = nn.Linear(embedding_dim, vocab_size)  #Embeddings of neighbour words

    def forward(self, inputs):
        embeds = self.embedding_w_c(inputs)
        out = self.embedding_u_o(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def embeddings(self):
        return self.embedding_w_c


loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 1000
for epoch in range(n_epochs):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        target_ix = word_to_ix[target]
        loss = loss_function(log_probs, torch.tensor([target_ix, target_ix], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    print (epoch, total_loss) #Loss should be decreasing


#Get embedding matrix
embedding = model.embeddings()


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, embedding):
    min_dist = 10000 # to act like positive infinity
    min_index = -1

    query_vector = embedding(torch.tensor(word_index)).detach().numpy()

    for index in range(len(vocab)):

        vector = embedding(torch.tensor(index)).detach().numpy()
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):

            min_dist = euclidean_dist(vector, query_vector)
            min_index = index

    return min_index

print("Closest to and: ", ix_to_word[find_closest(word_to_ix['and'], embedding)])
print("Closest to be: ", ix_to_word[find_closest(word_to_ix['be'], embedding)])
print("Closest to my:", ix_to_word[find_closest(word_to_ix['my'], embedding)])
