import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer, EnglishStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

example_string = """
... Muad'Dib learned rapidly because his first training was in how to learn.
... And the first lesson of all was the basic trust that he could learn.
... It's shocking to find how many people do not believe they can learn,
... and how many more believe learning to be difficult."""

sent_tok = sent_tokenize(example_string, "english")
# print(sent_tok)
word_tok = word_tokenize(example_string)
# print(word_tok)

worf_quote = "Sir, I protest. I am not a merry man!"
word_in_quote = word_tokenize(worf_quote)
# print(word_in_quote)
english_stp = stopwords.words("english")
# print([word for word in word_in_quote if word.casefold() not in english_stp])
# print(english_stp)
string_for_stemming = """
... The crew of the USS Discovery discovered many discoveries.
... Discovering is what explorers do."""
token_words = word_tokenize(string_for_stemming)
# print(token_words)
stemmer = SnowballStemmer("english")
eng_stemmer = EnglishStemmer(ignore_stopwords=False)
# print([eng_stemmer.stem(word) for word in token_words])

sagan_quote = """
... If you wish to make an apple pie from scratch,
... you must first invent the universe."""

new_token1 = word_tokenize(sagan_quote)
# print(nltk.pos_tag(new_token1))
# print(nltk.pos_tag(["hello", "!"]))
# print(nltk.help.upenn_tagset())

lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("starving", pos='v'))

lotr_quote = "It's a dangerous business, Frodo, going out your door."
# print(word_tokenize((lotr_quote)))
print(nltk.pos_tag(word_tokenize(lotr_quote)))

grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(nltk.pos_tag(word_tokenize(lotr_quote)))
# print(tree.draw())
# grammar1 = """
# ... Chunk: {<.*>+}
# ...        }<JJ>{"""
# chink_parser = nltk.RegexpParser(grammar1)
# tree1 = chink_parser.parse(nltk.pos_tag(word_tokenize(lotr_quote)))
# print(tree1.draw())