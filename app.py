#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup, redirect, url_for, session
import logging, io, os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import scipy
import pickle
from scripts import tabledef
from scripts import forms
from scripts import helpers
import stripe
import json

app = Flask(__name__)
app.config["DEBUG"] = True

app.secret_key = os.urandom(12)  # Generic key for dev purposes only

STRIPE_PUBLISHABLE_KEY = 'pk_test_OrDBLnozM58BIipUx4XP30yI00sP2NEixe'
STRIPE_SECRET_KEY = 'sk_test_rcnLdbqxMdCz69gU6IjOyIUK00HT3ODaaN'

stripe.api_key = STRIPE_SECRET_KEY

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))


@app.route('/payment', methods=['POST'])
def payment_proceed():
    # Amount in cents
    amount = 25000

    customer = stripe.Customer.create(
        email=request.form['stripeEmail'],
        source=request.form['stripeToken']
    )

    charge = stripe.Charge.create(
        amount=amount,
        currency='usd',
        customer=customer.id,
        description='A payment for the Hello World project'
    )

    return render_template('wine.html')

# model related variables
gbm_model = None
features = ['fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol',
            'color']



def get_wine_image_to_show(wine_color, wine_quality):
    if wine_color == 0:
        wine_color_str = 'white'
    else:
        wine_color_str = 'red'
    return('/static/images/wine_' + wine_color_str + '_' + str(wine_quality) + '.jpg')

@app.before_first_request
def startup():
    global gbm_model

    # load saved model from web app root directory
    gbm_model = pickle.load(open("static/pickles/gbm_model_dump.p", 'rb'))


@app.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@app.route('/background_process', methods=['POST', 'GET'])
def background_process():
    fixed_acidity = float(request.args.get('fixed_acidity'))
    volatile_acidity = float(request.args.get('volatile_acidity'))
    citric_acid = float(request.args.get('citric_acid'))
    residual_sugar = float(request.args.get('residual_sugar'))
    chlorides = float(request.args.get('chlorides'))
    free_sulfur_dioxide = float(request.args.get('free_sulfur_dioxide'))
    total_sulfur_dioxide = float(request.args.get('total_sulfur_dioxide'))
    density = float(request.args.get('density'))
    pH = float(request.args.get('pH'))
    sulphates = float(request.args.get('sulphates'))
    alcohol = float(request.args.get('alcohol'))
    color = int(request.args.get('color'))


    # create data set of new data
    x_test_tmp = pd.DataFrame([[fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        color]], columns = features)


    # predict quality based on incoming values
    preds = gbm_model.predict_proba(x_test_tmp[features])

    # get best quality prediction from original quality scale
    predicted_quality = [3,6,9][np.argmax(preds[0])]
    return jsonify({'quality_prediction':predicted_quality, 'image_name': get_wine_image_to_show(color, predicted_quality)})


@app.route("/wine", methods=['POST', 'GET'])
def show_wine():
    # on load set form with defaults
    return render_template('wine.html', quality_prediction=1, image_name='/static/images/wine_red_6.jpg')

# when running app locally
if __name__=='__main__':
    app.run(debug=True)