from flask import redirect, render_template, flash, url_for, request
from model import app
from model.classical_model import plot_roc, preds, plot_graphs
import json

@app.route('/')
def home():
    _, _ = plot_roc(preds)
    with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
        data = json.load(f)
    mse = data[1]['evals']['stock_mse']
    return render_template('Home.html', mse=mse)

@app.route('/documentation')
def doc():
    return render_template('doc.html')