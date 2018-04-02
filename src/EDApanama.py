stat = 'ACTIVE'
stat_country= 'Saudi Arabia'

mask1 = intermediaries['status'].str.contains(stat)
mask2 = intermediaries['countries'].str.contains(stat_country)

stage = intermediaries[mask1 & mask2]
stage


C = entities['countries'].value_counts()
C[:25].plot(kind='bar')

##########
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

f, axarr = plt.subplots(2, 2, figsize=(12, 11))
f.subplots_adjust(hspace=0.75)
plt.suptitle('Cash-Stash Entity Breakdown', fontsize=18)

entities['sourceID'].value_counts().plot.bar(ax=axarr[0][0])
axarr[0][0].set_title("Data Source")

entities['service_provider'].value_counts(dropna=False).plot.bar(ax=axarr[0][1])
axarr[0][1].set_title("Money Manager")

entities['jurisdiction_description'].value_counts().head(10).plot.bar(ax=axarr[1][0])
axarr[1][0].set_title("Jurisdiction (n=10)")

entities['countries'].value_counts(dropna=False).head(10).plot.bar(ax=axarr[1][1])
axarr[1][1].set_title("Home Country (n=10)")

Sa = all_nodes.loc[all_nodes['countries']=='Saudi Arabia'].copy()
Sa['service_provider'].value_counts().plot(kind='pie',autopct='%1.0f%%'
