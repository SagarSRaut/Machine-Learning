import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Housing.csv")

ohc = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_include=['int64',
                                            'float64'])))
dum_np = ct.fit_transform(df)

print(ct.get_feature_names_out())

col_names = ct.named_transformers_["onehotencoder"].get_feature_names_out().tolist() +\
    df.columns[df.dtypes=='float64'].tolist() +\
        df.columns[df.dtypes=='int64'].tolist()
dum_np_pd = pd.DataFrame(dum_np,columns=col_names)            

###########################################
ohc = OneHotEncoder(sparse_output=False, drop='first')
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_include=['int64',
        'float64'])), 
       verbose_feature_names_out=False ).set_output(transform='pandas')
dum_pd = ct.fit_transform(df)


