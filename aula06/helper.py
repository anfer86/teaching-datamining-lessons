import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def countplot_sex_random_variable_survived_titanic():
    df = sns.load_dataset('titanic')
    fig, ax = plt.subplots(1,2, figsize=(14,6))
    sns.countplot(x = 'sex', hue = 'survived', dodge = True, data = df, ax = ax[0])
    ax[0].set_title('Sex vs Survived')
    random_variable = np.random.choice(a=['A','B','C'], size=len(df), replace = True)
    ax[1].set_title('Random_Variable vs Survived')
    sns.countplot(x = random_variable, hue = 'survived', dodge = True, data = df, ax = ax[1])
    return fig, ax