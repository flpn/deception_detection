import matplotlib.pyplot as plt


labels = ('half-true', 'false', 'mostly-true','barely-true', 'true', 'pants-fire')
sizes = dataset['Label'].value_counts().values
explode = (0, 0.2, 0, 0)
fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('pie_shades_of_thruth.png')
plt.show()