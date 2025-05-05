def preprocess_function(examples, tokenizer):
    pad_on_right = tokenizer.padding_side == "right"
    model_inputs = tokenizer(
        examples["answers" if pad_on_right else "contexts"],
        examples["contexts" if pad_on_right else "answers"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=1024,
        padding="max_length")
    labels = tokenizer(
        examples["questions"], max_length=256, truncation=True, padding=True
    )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs

def preprocess_function_without_answer(examples, tokenizer):
    model_inputs = tokenizer(
        examples["contexts"], max_length=1024, truncation=True, padding=True
    )
    labels = tokenizer(
        examples["questions"], max_length=256, truncation=True, padding=True
    )

    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs