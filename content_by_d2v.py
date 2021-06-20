# -*- coding:utf-8 -*-

# python example to train doc2vec model (with or without pre-trained word embeddings)

import smart_open
import gensim
import logging

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def gensim_models_doc2vec():
    # Set file names for train and test data
    all_corpus_file = "data/cmu/train_corpus/cmu_all_process.txt"
    all_corpus = list(read_corpus(all_corpus_file))

    # get model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=128, min_count=2, epochs=40, window=10, sample=1e-3,
                                          negative=5, workers=3, hs=0, ns_exponent=0.75, dm=0)
    print("start build ...")
    model.build_vocab(documents=all_corpus, progress_per=10000, keep_raw_vocab=False)
    print("start train ...")
    model.train(documents=all_corpus, total_examples=model.corpus_count, epochs=model.epochs, word_count=0)
    model.init_sims(replace=True)

    # save model
    model.save("./data/cmu/train_corpus/model_dim_128_epoch_40.bin")


def preprocess():
    # Import and download stopwords from NLTK.
    from nltk.corpus import stopwords
    # from nltk import download
    # download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')
    interpunction = "||| 《 》 # @ ' —— + - . ! ！ / _ , $ ￥ % ^ * ( ) ] | [ ， 。 ？ ? : ： ； ; 、 ~ …… … & * （ ）".split()

    with open("data/cmu/train_corpus/cmu_all.txt", 'r') as f_in:
        lines = f_in.readlines()
    f_in.close()
    total_line = len(lines)
    print("total line is:{}".format(total_line))

    str_out = " "
    for index, line in enumerate(lines):
        line = line.lower().split()
        line = [w for w in line if w not in interpunction]  # Remove interpunction.
        line = [w for w in line if w not in stop_words]  # Remove stopwords.
        line = ' '.join(line) + "\n"
        str_out = str_out + line
        if (index + 1) % 1000 == 0 or index == (total_line - 1):
            with open("data/cmu/train_corpus/cmu_all_process.txt", 'a') as f_out:
                f_out.write(str_out)
            print("preprocess num {} , and average of word is {}".format(index + 1, len(str_out.split()) / 1000))
            str_out = " "
    f_out.close()

    # verify the line num of files.
    with open("data/cmu/train_corpus/cmu_all_process.txt", 'r') as f_in:
        lines_fin = f_in.readlines()
        print("total line is:{}".format(len(lines_fin)))
    f_in.close()


if __name__ == '__main__':
    preprocess()
    gensim_models_doc2vec()
