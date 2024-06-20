from transformers import LlamaForCausalLM, LlamaTokenizer

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    # Load the tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained("yourusername/llama2-lora")
    model = LlamaForCausalLM.from_pretrained("yourusername/llama2-lora")

    # Encode the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    outputs = model.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time in a land far, far away,"
    generated_text = generate_text(prompt)
    print(generated_text)
