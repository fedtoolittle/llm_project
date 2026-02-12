# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.processors import TemplateProcessing

# # Initialize tokenizer
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# trainer = BpeTrainer(
#     vocab_size=8000,
#     special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
# )

# tokenizer.train(["train_text.txt"], trainer)

# # Add post-processing to automatically add BOS/EOS
# tokenizer.post_processor = TemplateProcessing(
#     single="[BOS] $A [EOS]",
#     special_tokens=[
#         ("[BOS]", tokenizer.token_to_id("[BOS]")),
#         ("[EOS]", tokenizer.token_to_id("[EOS]")),
#     ],
# )

# tokenizer.save("tokenizer.json")

# print("Tokenizer trained and saved.")


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE())

# IMPORTANT
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train(["wikitext103_train.txt"], trainer)

tokenizer.save("tokenizer.json")