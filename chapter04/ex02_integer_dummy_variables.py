import pandas as pd
from prettytable import PrettyTable


def display(df):
    table = PrettyTable(field_names=list(df.columns))
    for row in df.values:
        table.add_row(row)
    print table
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})

display(demo_df)
dummies = pd.get_dummies(demo_df)
display(dummies)
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
display(demo_df)
dummies = pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
display(dummies)
print "data now with raw type: \n%r" % demo_df.values
