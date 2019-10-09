

def confusion_matrix_df(confusion_matrix):
    return pd.DataFrame(data = confusion_matrix, 
                        index = ['observed_negative','observed_positive'], 
                        columns = ['predicted_negative','predicted_positive'])
