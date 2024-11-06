from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

unmasker = pipeline('fill-mask', 
                    model=AutoModelForCausalLM.from_pretrained('results/'),
                    tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                    top_k=10)

r=unmasker("The [MASK] is at a tipping point.")



from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("results/")#, tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'))

# The sentences to encode
sentences = [
"Time is a thief, stealing moments away.",
"The days seem to be passing quickly."
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

