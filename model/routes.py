from flask import redirect, render_template, flash, url_for, request
from model import app
from model.classical_model import plot_roc, preds, plot_graphs

@app.route('/')
def home():
    _, _ = plot_roc(preds)
    return render_template('Home.html')

@app.route('/documentation')
def doc():
    return render_template('doc.html')