from util import get_llm, tokenizer

# Generate text based on a prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
model = get_llm()
output = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
