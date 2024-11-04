from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
r=unmasker("The [MASK] is at a tipping point.")

r[len(r)-1]