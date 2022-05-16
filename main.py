#AI Project
# Pie no1
results_df = pd.DataFrame({'Labels': Test_Labels,
                           'Prediction': y_pred})

s_list = []

for i in range(9):
    if y_pred[i] == Test_Labels[i]:
        s_list.append('Success')
    else:
        s_list.append('Fail')

results_df['Outcome'] = s_list
results_df.plot.pie(y=s_list)

# Pie no2
fig1, ax = pyplot.subplots()
patches, text, auto = ax.pie(s_list, autopct='%1.1f%%', colors=['red','green'],
                             wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                             pctdistance=1.15, startangle=160)
circle = pyplot.Circle((0, 0), 0.5, color='white')
pyplot.gcf().gca().add_artist(circle)
pyplot.legend(patches, ['Success', 'Fail'], loc='upper right', fontsize='xx-small', framealpha=0.4)