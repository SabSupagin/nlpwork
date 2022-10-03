from flask import Flask,render_template,request
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
from gensim.models.tfidfmodel import TfidfModel
from werkzeug.utils import secure_filename
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from flaskext.markdown import Markdown

import itertools
import os
import nltk

import filecal.ner as n

# nltk.download('punkt')
# nltk.download('omw-1.4')

app = Flask(__name__)
Markdown(app)

@app.route('/')
def hello():
    return render_template('show.html')


patch = os.getcwd() + "\\files"
# save_patch = []

@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/bowwithtf')
def bowtf():
    return render_template('bowwithtf.html')


@app.route('/show', methods=['POST'])
def showPro():
    # get data normaal
    # data = request.form.get('files')

    file_list = request.files.getlist('files')
    for f in file_list:
        join_patch = os.path.join(patch, secure_filename(f.filename))
        f.save(join_patch)
        # save file
        # save_patch.append(join_patch)


    return render_template('show.html')


@app.route('/upload', methods=['POST'])
def upload():

    # ทำการลบไฟล์ในโฟลเดอร์ทั้งหมดเมื่อมีการจะ upload ไฟล์ใหม่
    if request.method == 'POST':
        dir_path = r'./files/'
        count = 0
       
        for path in os.listdir(dir_path):
        
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1

    for i in range(count):
        os.remove(f"./files/wiki_article_{i}.txt")
        # for i in save_patch:
        #     os.remove(i)

    return render_template('upload.html')

#คำนวณ

#ค้นหาคำที่ต้องการ

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        word = request.form['word']
        dir_path = r'./files/'
        count = 0
       
        for path in os.listdir(dir_path):
        
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1

    #แปลงคำในไฟล์เป็นโทเคน          
     
    articles = []
    for i in range(count):
        f = open(f"./files/wiki_article_{i}.txt", "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    # for i in save_patch:
    #     f = open(i,"r")
    #     article = f.read()
    #     tokens = word_tokenize(article)
    #     lower_tokens = [t.lower() for t in tokens]
    #     alpha_only = [t for t in lower_tokens if t.isalpha()]
    #     no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
    #     wordnet_lemmatizer = WordNetLemmatizer()
    #     lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    #     articles.append(lemmatized)

    #ค้นหาคำที่ต้องการ

    dictionary = Dictionary(articles)
    word_id = dictionary.token2id.get(word)
    showword = ("Dictionary : "+word+' found = '+str(word_id))
    return render_template('show.html',showword = showword)

#5อันดับคำที่มีมากที่สุด
@app.route('/topword', methods = ['GET', 'POST'])
def topword():

    if request.method == 'POST':
        dir_path = r'./files/'
        count = 0
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1

    #แปลงคำในไฟล์เป็นโทเคน

    articles = []
    for i in range(count):
        f = open(f"./files/wiki_article_{i}.txt", "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    
    # for i in save_patch:
    #     f = open(i,"r")
    #     article = f.read()
    #     tokens = word_tokenize(article)
    #     lower_tokens = [t.lower() for t in tokens]
    #     alpha_only = [t for t in lower_tokens if t.isalpha()]
    #     no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
    #     wordnet_lemmatizer = WordNetLemmatizer()
    #     lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    #     articles.append(lemmatized)

    # แบบ BOW

    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    doc = corpus[0]
    bowshow =[]
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
    for word_id, word_count in sorted_word_count[0:5]:
        b = (dictionary.get(word_id), word_count)
        bowshow.append(b)

    # แบบ TF-IDF

    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
    topshow = []

    #วนลูปหา top 5 ของทุกๆ text file

    for term_id, weight in sorted_tfidf_weights[:5]:
        a = dictionary.get(term_id), weight
        topshow.append(a)
    return render_template('bowwithtf.html', topshow = topshow, bowshow = bowshow)

#fake or re
@app.route('/fakenews', methods=['POST'])
def fakenews():
    if request.method == 'POST':
        word = request.form['forr']
        model_path = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_length = 512

        def get_prediction(text, convert_to_label=False):
            inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length,return_tensors="pt")
            outputs = model(**inputs)
            probs = outputs[0].softmax(1)
            d = {
            0: "reliable",
            1: "fake"
            }
            if convert_to_label:
                return d[int(probs.argmax())]
            else:
                return int(probs.argmax())
        real_news = word
        fakeask = get_prediction(real_news, convert_to_label=True)

    return render_template('show.html', fakeask = fakeask)

@app.route('/nerpage', methods = ['POST'])
def ner_page():
    return render_template('ner.html')

@app.route('/ner', methods = ['POST'])
def ner_process():
    if request.method == 'POST':
        inp = request.form.get('filetext')
        tex = n.ner_processing(inp)
    return render_template('ner.html',ask=tex)

if __name__ == '__main__':
    app.run(debug=True)
