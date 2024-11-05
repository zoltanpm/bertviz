import openai
import json
import polars as pl

client = openai.OpenAI()

completion = client.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=[
        {"role": "system", "content": "You're skilled at coming up with metaphors."},
        {"role":"user", "content": "Create 500 unique examples of sentences like the ones below, each containing a metaphor. \
            Format your output as a JSON that's clean and can be parsed with the Python 'json.loads' function. Do not append anything to the sentence.\
            Here are some examples: \
            'The climate is at a tipping point. \
            'I could eat a horse. \
            'These are the dog days of summer. \
            'I'm going to hit the books.'"}            
            ],
    response_format={"type": "json_object"}
)


data = json.loads(completion.choices[0].message.content)
df = pl.DataFrame(data)

df.write_csv('synthetic_metaphors.csv')
