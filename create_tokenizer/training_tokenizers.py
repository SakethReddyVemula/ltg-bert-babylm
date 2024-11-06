# Import the tokenizer and subword BPE trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer

## Pre-tokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace

unk_token = "[UNK]"
spl_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]", "[TAB]"]

def prepare_tokenizer_trainer(alg, vocab_size=50227, min_frequency=2):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token = unk_token))
        trainer = BpeTrainer(special_tokens = spl_tokens, vocab_size = vocab_size, min_frequency=min_frequency)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token = unk_token, special_tokens = spl_tokens, vocab_size = vocab_size, min_frequency=min_frequency)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
        trainer = WordPieceTrainer(special_tokens = spl_tokens, vocab_size = vocab_size, min_frequency=min_frequency)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token = unk_token))
        trainer = WordLevelTrainer(special_tokens = spl_tokens, vocab_size = vocab_size, min_frequency=min_frequency)
    
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(files, alg='WLV', vocab_size=50227, save_path="./tokenizer-trained.json"):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, vocab_size=vocab_size)
    tokenizer.train(files, trainer) # training the tokenzier
    tokenizer.save(save_path)
    tokenizer = Tokenizer.from_file(save_path)
    return tokenizer

corpus = ["/media/saketh/New Volume/NAACL 2025/Datasets/en/en_10M_splits_morphed.txt"]
trained_tokenizer = train_tokenizer(corpus, "BPE", vocab_size=16384, save_path="./en/en_M-BPE_16384.json")
# input_string = "மேலும் அறியப்படாத எதிர்காலத்தைப் பற்றி நன்றாக உணர தங்கள் சொந்த பொய்களை முழுமையாக நம்புவார்கள்."
# input_string = "The government's liberalised policy on permitting foreign collaborations takes in collaborations for discotheques as well."
# input_string = "స్కిన్ డిసార్డర్స్: స్థానికంగా దరఖాస్తు, బాసిల్ రసం రింగ్వార్మ్ మరియు ఇతర చర్మ వ్యాధుల చికిత్సలో ఉపయోగకరంగా ఉంటుంది."
# input_string = "जानकारी के लिए आपको बता दें कि 24 अक्टूबर को महाराष्ट्र विधानसभा चुनाव के नतीजे जारी किए गए।"
input_string = "All participants were tested and diagnosed as being free of any cardiovascular disease at the time of the baseline s can."

encoded = trained_tokenizer.encode(input_string)
tokens = encoded.tokens
ids = encoded.ids
print("Tokens:", tokens)
print("IDs:", ids)