
from __future__ import division, print_function
# coding=utf-8
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# Pickle
import pickle

# Flask utils
from flask import Flask, flash, redirect, request, render_template
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from fbprophet import Prophet
# Define a flask app
app = Flask(__name__)

# Model saved
MODEL_PATH ='spi.pkl'

# Load your trained model
model = pickle.load(open(MODEL_PATH, 'rb'))



def model_predict(img_path, model):
    
    
    file = file_path + '/' +"HISTWIDE_LOWES.csv"
    out_file_path = os.path.join(basepath, 'output')
    data=pd.read_csv(file)
    data.columns = data.columns.astype(str).str.replace('PERIOD', '')
    data['STARTDATE'] = pd.to_datetime(data['STARTDATE'])
    data.drop(['DMDGROUP','EVENT','HISTSTREAM','DMDCAL', 'TYPE'], axis=1, inplace=True)
    
    tData=data.melt(id_vars=['DMDUNIT','LOC','STARTDATE'], var_name='WEEK', value_name='SALES')
    tData['WEEK']=tData['WEEK'].astype(int)
    temp = tData['WEEK'].apply(lambda x: pd.Timedelta(x, unit='W'))
    tData['Date'] = tData['STARTDATE'] + (temp)
    tData['SALES']=tData['SALES'].str.replace(',', '')
    tData.dropna(subset = ["SALES"], inplace=True)
    tData['SALES']=tData['SALES'].astype(float)
    uniqueStore=tData.LOC.unique()
    GData = tData.groupby('LOC')
    tcData = tData.copy()

    output = []
    tData = tcData.copy()
    for i in range(0, len(uniqueStore)):
        uStore = tData.loc[tData['LOC'] == uniqueStore[i]]
        uStore['Normalized']=(uStore['SALES']-uStore['SALES'].min())/(uStore['SALES'].max()-uStore['SALES'].min())    
        output.append(uStore)
    
    output2 = pd.DataFrame()
    for i in range(0,len(output)):
        output2 = output2.append(pd.DataFrame(output[i]))
    output2.to_csv(out_file_path+ '/' + r'SPI_final_data.csv')    
    
    output3=output2
    output4=output2

    output5=output2
    output6=output2
    output7=output2
    
    output3 = output3.loc[output3['LOC'] == 'G1010_L0']
    output4 = output4.loc[output4['LOC'] == 'G1010_L1']
    output5 = output5.loc[output5['LOC'] == 'G1249_L0']
    output6 = output6.loc[output6['LOC'] == 'G1249_L1']
    
    for i in range(0,104):
        output2["D"+str(i)] = np.where(output2['WEEK']== (i+1), 1, 0)
    
    output2 = output2.loc[output2['WEEK'] == 1]
    output2.dropna()
    output2.to_csv(out_file_path+ '/' + r'SPI_final_dummy_data.csv')
    
    data1 = output2.iloc[:, 6].values
    data2=pd.DataFrame(data1)
    data2.dropna()
    data2[0] = data2[0].fillna(0)
    
    pred=KMeans(n_clusters=10).fit_predict(data2)
    Y=pred
    data2['Profiles'] = Y
    output2['Profiles']= Y
    output2.to_csv(out_file_path+ '/' + r'SPI_final_dummy_data_groups.csv')
    Groups = output2.groupby('Profiles')['LOC'].count()
    
    output3 = output3.loc[output3['LOC'] == 'G1010_L0']
    output4 = output4.loc[output4['LOC'] == 'G1010_L1']
    output5 = output5.loc[output5['LOC'] == 'G1249_L0']
    output6 = output6.loc[output6['LOC'] == 'G1249_L1']
    output7 = output7.loc[output7['LOC'] == 'G1683_L1']
    
    plt.figure(figsize=(20, 8))
    plt.plot(output3['Date'], output3['SALES'], 'b-', label = 'G1010_L0')
    plt.plot(output4['Date'], output4['SALES'], 'r-', label = 'G1010_L1')
    plt.plot(output5['Date'], output5['SALES'], 'g-', label = 'G1249_L0')
    plt.plot(output6['Date'], output6['SALES'], 'y-', label = 'G1249_L1')
    plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of G1010 Vs G1249')
    plt.legend();
    plt.savefig(out_file_path+ '/' + 'Sales of G1010 Vs G1249.png')
    
    plt.figure(figsize=(20, 8))
    plt.plot(output7['Date'], output7['SALES'], 'o-', label = 'G1683_L2')
    plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('G1683_L2')
    plt.legend();
    plt.savefig(out_file_path+ '/' + 'Sales of G1683_L2.png')
    
    output7 = output7.rename(columns={'Date': 'ds', 'SALES': 'y'})
    output7['ds']=pd.to_datetime(output7['ds'])
    
    G1683_L2_model = Prophet()
    G1683_L2_model.fit(output7)
    
    G1683_L2_forecast = G1683_L2_model.make_future_dataframe(periods=52, freq='W')
    G1683_L2_forecast = G1683_L2_model.predict(G1683_L2_forecast)
    
    plt.figure(figsize=(20, 8))
    G1683_L2_model.plot(G1683_L2_forecast, xlabel = 'Date', ylabel = 'Sales')
    plt.title('G1683 Sales')
    plt.savefig(out_file_path+ '/' + 'G1683 Sales')
    
    plt.figure(figsize=(20, 8))
    output7['y'].plot()
    plt.savefig(out_file_path+ '/' + 'output7.png')
    
    future_dates=G1683_L2_model.make_future_dataframe(periods=52,freq='W')
    prediction=G1683_L2_model.predict(future_dates)
    prediction.to_csv(out_file_path+ '/' + r'prediction.csv')
    
    #### plot the predicted projection
    G1683_L2_model.plot(prediction)      
    plt.savefig(out_file_path+ '/' + 'prediction.png')    
    ##### Visualize Each Components[Trends,Weekly]
    G1683_L2_model.plot_components(prediction)
    plt.savefig(out_file_path+ '/' + 'Trends Weekly.png')
    
    from fbprophet.diagnostics import cross_validation
    output7_cv=cross_validation(G1683_L2_model,horizon="365 days",period='180 days',initial='60 days')
    
    from fbprophet.diagnostics import performance_metrics
    df_performance=performance_metrics(output7_cv)
    df_performance.head()
    
    from fbprophet.plot import plot_cross_validation_metric
    fig=plot_cross_validation_metric(output7_cv,metric='mdape')
    plt.savefig(out_file_path+ '/' + 'cross validation metrics.png')
    

    
    return Groups

basepath = os.path.dirname(__file__)
file_path = os.path.join(basepath, 'uploads')
UPLOAD_FOLDER = file_path
app.secret_key = "my-secret-Balla"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xls','xlsx'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def upload_form():
 return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():

        
        if request.method == 'POST':
        # check if the post request has the files part
            if 'files[]' not in request.files:
                flash('No file part')
                return redirect(request.url)
            files = request.files.getlist('files[]')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #flash('File(s) successfully uploaded')
                       
        
        # Make prediction
        Groups = model_predict(file_path, model)
        Groups.to_csv('Groups.csv')
        data = pd.read_csv('Groups.csv')
        return data.to_html()


if __name__ == '__main__':
    app.run(debug=True)
