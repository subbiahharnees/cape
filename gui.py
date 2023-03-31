from flask import Flask, request, render_template
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    with open('viewmodel.html', 'r') as mdll:
        mdl = mdll.read()
        return mdl


@app.route('/svm_model', methods=['POST'])
def svm_model():
    import numpy as np
    feat = request.form['values']
    type = ['Commercial', 'Military', 'Training', 'Executives', 'Others',
            'Financing', 'Support & Services']
    raw_data = np.load('pre_evaluated/data.npy', allow_pickle=True)[int(feat)]
    pre_data = np.load('pre_evaluated/pre_data.npy', allow_pickle=True)[int(feat)]
    ftt = np.load('pre_evaluated/vect_data.npy', allow_pickle=True)[int(feat)]
    ft = ftt.reshape(1, -1)
    # ft = [float(num) for num in feat.split(",")]
    import tensorflow
    # from tensorflow.keras import models
    # cn_mdl = models.load_model('mdl.h5')
    svc_mdl = joblib.load('svm_model.pkl')
    pred = svc_mdl.predict(ft)[0]  # (svc_mdl.predict_classes(ft)).flatten()
    rslt = [raw_data, pre_data, ftt, type[pred]]
    return render_template('result.html', result=rslt)


# if __name__ == '__main__':  # to run separate
#     app.run()

def guii():
    app.run()
