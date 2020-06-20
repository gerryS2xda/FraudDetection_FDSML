from getdata import fetchdata
from matplotlib import pyplot as plt


#Call function
print('Loading data...')
fraudata = fetchdata.load_fraud_data()

print('Head:')
print(fraudata.head(10)) #necessario perche' PyCharm non e' Python Shell
print('Info:')
fraudata.info(null_counts=True) #
print('Categorical:')
print(fraudata["type"].value_counts())
print('Describe:')
print(fraudata.describe())
print('Istogrammi:')

fraudata.plot(kind='bar')




