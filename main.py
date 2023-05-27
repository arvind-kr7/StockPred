from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np

from datetime import datetime
import datetime as dt
import yfinance as yf
from utils import Backend


#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/', methods=['GET', "POST"])
def index():
   print(request.method)
   context= {'error': False}
   
   
   if request.method == 'POST':
      
      company = request.values.get('company')
      backend = Backend(company)
      res = backend.predict()
      if res:
        context={'result_src':f'static/prediction_{company}.png', 'company':company}
        return render_template('result.html', **context)
      
      else:
        context['error']=f'invalid kerword: {company}'
      
   else:
      return render_template('index.html', **context)



if __name__ == '__main__':
   app.run()

