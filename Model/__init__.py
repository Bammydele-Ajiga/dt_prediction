# import matplotlib.pyplot as plt
import pickle
import pandas as pd
filename1 = "Model/scaler.sav"
scaler = pickle.load(open(filename1, 'rb'))
scaler1 = pickle.load(open(filename1, 'rb'))
filename = "Model/p_sonic_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
def get_pred_dt_values(filepath):
  file = open(filepath, "r")
  well = pd.read_csv(file)
  logs = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI',"DEPTH"]
  depth = well["DEPTH"]
  well = well[logs]
  features = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI']
  well_features = well[features]
  well_features = scaler.fit_transform(well_features)
  dT_pred = loaded_model.predict(well_features)
  dT_pred =  scaler1.inverse_transform(dT_pred.reshape(-1,1))
  dT_pred1= pd.DataFrame(dT_pred, columns= ["DT"])
  # f, ax = plt.subplots(figsize=(4,14))
  # ax.plot(dT_pred,depth , color="green")
  # ax.set_xlim([max(dT_pred), min(dT_pred)])
  # ax.set_ylim([depth.max(), depth.min()])
  result = pd.concat([well,dT_pred1], axis = 1)
  return result
#
# result = get_pred_dt_values("psonicTest.csv")
# print(result.head())
# print("hello world")