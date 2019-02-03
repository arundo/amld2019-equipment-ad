# example of app.py file

"""
To deploy your own serialized model

"""

from runtime.framework import endpoint, argument, returns
import numpy as np

# model dependent libraries
from sklearn.externals import joblib


# global variables can be loaded into memory to improve efficiency
# be careful if using tensorflow globally
# update file names according to your own models, as needed

scaler_filename = 'scaler.model'
model_filename = 'RandomForestClassifier.model'
G_SCALER = joblib.load(scaler_filename)
MODEL = joblib.load(model_filename)


# update the argument names below based on the final list of features
# that you have used in your model. 


@endpoint()
@argument("setting1", type=float, description="Value of input sensor setting1")
@argument("setting2", type=float, description="Value of input sensor setting2")
@argument("setting3", type=float, description="Value of input sensor setting3")
@argument("s1", type=float, description="Value of input sensor s1")
@argument("s2", type=float, description="Value of input sensor s2")
@argument("s3", type=float, description="Value of input sensor s3")
@argument("s4", type=float, description="Value of input sensor s4")
@argument("s5", type=float, description="Value of input sensor s5")
@argument("s6", type=float, description="Value of input sensor s6")
@argument("s7", type=float, description="Value of input sensor s7")
@argument("s8", type=float, description="Value of input sensor s8")
@argument("s9", type=float, description="Value of input sensor s9")
@argument("s10", type=float, description="Value of input sensor s10")
@argument("s11", type=float, description="Value of input sensor s11")
@argument("s12", type=float, description="Value of input sensor s12")
@argument("s13", type=float, description="Value of input sensor s13")
@argument("s14", type=float, description="Value of input sensor s14")
@argument("s15", type=float, description="Value of input sensor s15")
@argument("s16", type=float, description="Value of input sensor s16")
@argument("s17", type=float, description="Value of input sensor s17")
@argument("s18", type=float, description="Value of input sensor s18")
@argument("s19", type=float, description="Value of input sensor s19")
@argument("s20", type=float, description="Value of input sensor s20")
@argument("s21", type=float, description="Value of input sensor s21")
@returns("predicted_class", type=float,
         description="Prediction as a float: 1.0 if anomaly, 0.0 otherwise")
def anomaly_detection(
    setting1, setting2, setting3,
    s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8,  s9,  s10,
    s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21
    ):
    """
    This endpoint is used to connect a live stream of data to a model and
    return predictions in real time for each new feature row as it arrives
    from a remote location.
    """

    # take the inputs and put them into and ordered np.array shape (, 29)
    feature_list = np.array([
        setting1, setting2, setting3,
        s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8,  s9,  s10,
        s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21
    ])

    # apply the globally loaded feature scaler
    scaled = G_SCALER.transform(np.atleast_2d(feature_list))

    # perform prediction using the model
    yhat = MODEL.predict(scaled)

    return float(yhat)
