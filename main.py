# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:35:16 2019

@author: Vishvaas
"""

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField


import os
import sys
import pickle
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructType, StructField, StringType, FloatType

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


def process(name):
    CLOUDSQL_INSTANCE_IP = ''   #(database server IP)
    CLOUDSQL_DB_NAME = 'recommendation_spark'
    CLOUDSQL_USER = 'root'
    CLOUDSQL_PWD  = 'tiger'  # CE
    
    conf = SparkConf().setAppName("train_model")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    USER_ID=name
    jdbcDriver = 'com.mysql.jdbc.Driver'
    jdbcUrl    = 'jdbc:mysql://%s:3306/%s?user=%s&password=%s' % (CLOUDSQL_INSTANCE_IP, CLOUDSQL_DB_NAME, CLOUDSQL_USER, CLOUDSQL_PWD)
    
    # checkpointing helps prevent stack overflow errors
    sc.setCheckpointDir('checkpoint/')
    
    # Read the ratings and accommodations data from Cloud SQL
    dfRates = sqlContext.read.format('jdbc').options(driver=jdbcDriver, url=jdbcUrl, dbtable='Rating', useSSL='false').load()
    dfAccos = sqlContext.read.format('jdbc').options(driver=jdbcDriver, url=jdbcUrl, dbtable='Accommodation', useSSL='false').load()
    print("read ...")
    
    # train the model
    model = ALS.train(dfRates.rdd, 20, 20) # tuning number
    print("trained ...")
    
    # use this model to predict what the user would rate accommodations that she has not rated
    allPredictions = None
    dfUserRatings = dfRates.filter(dfRates.userId == USER_ID).rdd.map(lambda r: r.accoId).collect()
    rddPotential  = dfAccos.rdd.filter(lambda x: x[0] not in dfUserRatings)
    pairsPotential = rddPotential.map(lambda x: (USER_ID, x[0]))
    predictions = model.predictAll(pairsPotential).map(lambda p: (str(p[0]), str(p[1]), float(p[2])))
    predictions = predictions.takeOrdered(5, key=lambda x: -x[2]) # top 5
    print("predicted for user={0}".format(USER_ID))
    if (allPredictions == None):
     allPredictions = predictions
    else:
     allPredictions.extend(predictions)
    
    # write them
    schema = StructType([StructField("userId", StringType(), True), StructField("accoId", StringType(), True), StructField("prediction", FloatType(), True)])
    dfToSave = sqlContext.createDataFrame(allPredictions, schema)
    dfToSave.write.jdbc(url=jdbcUrl, table='Recommendation', mode='overwrite')




class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
 
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print(form.errors)
    if request.method == 'POST':
        name=request.form['name']
        print(name)
 
    if form.validate():
        process(name)
        
    else:
        flash('Error: All the form fields are required. ')
 
    return render_template('hello.html', form=form)
 
    



if __name__ == "__main__":
    app.run()