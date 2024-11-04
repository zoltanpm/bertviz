from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased', top_k=10)
r=unmasker("The [MASK] is at a tipping point.")
r
r[len(r)-1]

