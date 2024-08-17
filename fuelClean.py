import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('cleanfuel.csv')
print(df['city_mpg_ft1'].unique())

bins = [2004, 2006, 2008, 2010, 2012, 2014, 2017]
labels = [0, 1, 2, 3, 4, 5]
#df['year'] = pd.cut(df['year'], bins=bins, labels=labels, include_lowest=True)
print(df['year'].unique())
#df.to_csv('cleanfuel.csv', index=False)



#string stuff
#encoder = LabelEncoder()
#encoder.fit(df['fuel_type'])
#df['fuel_type'] = encoder.transform(df['fuel_type'])
#df = df.drop(df[df.transmission == ].index)
#df = df[df['engine_displacement'] != 'nan']
#create bins that ranges fall under
