import os
from flask import Flask, render_template, request, url_for
from flask_assets import Environment, Bundle
import re
import requests
import pandas as pd
from tabulate import tabulate
from google_trans_new import google_translator
from transformers import pipeline
import matplotlib.pyplot as plt
import io 
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
from matplotlib.figure import Figure

classifier = pipeline('sentiment-analysis')

def cls_corpus_pipeline(corpus_list):
  data = []
  global df
  global df3
  for sentence in corpus_list:
    corpus_result = classifier(sentence)
    listToStr = ' '.join([str(elem) for elem in corpus_result])
    listToStr = listToStr.replace('\'', '')
    listToStr = listToStr.replace('}', '')
    listToStr = listToStr.replace(',', '')
    label = listToStr.split()[1]
    score = listToStr.split()[3]
    data.append([label, score, sentence])
    df = pd.DataFrame(data)

  df.columns=['Classification', 'Score', 'Text']
  class_count  = df['Classification'].value_counts().sort_index()
  s = df.Classification
  counts = s.value_counts()
  percent100 = s.value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
  df3 = pd.DataFrame({'Counts': counts, 'Percentage': percent100}).sort_index(ascending=False)

####################################################################################

app = Flask(__name__) 
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 10
app.debug = True

assets = Environment(app)

png = Bundle('plotSA.png')
assets.register('png_all', png)

@app.route('/', methods=['GET', 'POST'])
def index():
    #global dfPipelineCount
    errors = []
    results = {}
    reviews = {}
    if request.method == "POST":
        # get url that the user has entered
        try:
            url = request.form['url']
#            r = requests.get(url)
            print(url)
            numberInURL = re.findall('\d+', url)
            itemid = numberInURL[-1]
            shopid = numberInURL[-2]
            print("Shop ID: ", shopid)
            print("Item ID: ", itemid)

            ratings_url = 'https://shopee.com.my/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0'
            komen = []
            offset = 0

            while True:
                data = requests.get(ratings_url.format(shop_id=shopid, item_id=itemid, offset=offset)).json()

                i = 1
                for i, rating in enumerate(data['data']['ratings'], 1):
                    komen.append(rating['comment'])
                if i % 20:
                    break
                offset += 20

            komen = list(filter(None, komen))
            komen = [x.replace('\n', '') for x in komen]

            corpus_list = komen
            dataEN = []
            dataMSID = []
            for i in corpus_list: 
              if i != "":
                t = google_translator().detect(i)
                if t[0] == 'en':
                  dataEN.append([t[0], i])
                elif t[0] == 'ms' or t[0] == 'id':
                  dataMSID.append([t[0], i])

            dfKomenEN = pd.DataFrame(dataEN)
            dfKomenEN.columns = ['Language', 'Review']
            dfKomenMSID = pd.DataFrame(dataMSID)
            dfKomenMSID.columns = ['Language', 'Review']

            corpus = dfKomenEN['Review']
            corpus_list = corpus.tolist()

            cls_corpus_pipeline(corpus_list)
            dfPipeline = df.copy()
            dfPipelineCount = df3.copy()
            dfPipelineCount.sort_index(ascending=False, inplace=True)
            dfPipelineCount.reset_index(inplace=True)
            dfPipelineCount = dfPipelineCount.rename(columns = {'index':'Classification'})
            results = dfPipelineCount.values.tolist()
            reviews = dfPipeline.values.tolist()

            print("=======================================")
            print("Total Number of EN Reviews: ", dfPipeline['Classification'].count())
            print("1. Model: Pipeline: \n", dfPipelineCount)
            print("")

#            print("")
#            print("1. EN Model Pipeline: \n", tabulate(dfPipeline, showindex=False, headers=dfPipeline.columns))

            SMALL_SIZE = 12
            MEDIUM_SIZE = 14
            BIGGER_SIZE = 16

            plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
            #plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
            #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            #plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            #plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            colors_2 = ['#348017', '#C11B17']
            colors_3 = ['#348017', '#FFA62F', '#C11B17']
#            print("Sentiment Analysis on English reviews")
            plotSA = df3.plot.pie(y='Counts', colors = colors_2, autopct='%1.2f%%')

            fig = plotSA.get_figure()
            #fig.savefig("{0}.png".format(itemid))
            fig.savefig("static/plotSA.png")
 
        except:
            errors.append(
                "Unable to get the URL or there's no review for this product yet. Please make sure it's valid and try again."
            )
    return render_template('index.html', errors=errors, results=results, reviews=reviews)

#@app.route('/sentiment/<filename>')
#def sentiment(filename):
#    x = dfPipelineCount
#    return render_template("analysis.html", data=x.to_html())

#@app.route('/analysis/<filename>')
#def analysis(filename):
#    x = dfPipelineCount
#    return render_template("analysis.html", name=filename, data=x.to_html())

####################################################################################

if __name__ == '__main__':
    app.run(debug=True)
