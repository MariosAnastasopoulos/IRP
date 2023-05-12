


import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv('C:/Users/Marios/Documents/Astronautics and Space Engineering MSc/IRP-Hellas-Sat/HS3 Data/v2_HS3_RWA_Rates_H_2022.txt')

# convert date & time 
df['End time'] = pd.to_datetime(df['End time'], format='%d/%m/%Y %H:%M:%S.%f')
df['Start time'] = pd.to_datetime(df['Start time'], format='%d/%m/%Y %H:%M:%S.%f')
print(df.head())

#filter data by parameter 
sc_mntm_x=df[df['Parameter']== 'AW0004R']
sc_mntm_y=df[df['Parameter']== 'AW0005R']
sc_mntm_z=df[df['Parameter']== 'AW0006R']
sc_mntm_tot=df[df['Parameter']== 'HA0677D']
rw1_spd=df[df['Parameter']== 'AW1010R']
rw2_spd=df[df['Parameter']== 'AW2010R']
rw3_spd=df[df['Parameter']== 'AW3010R']
rw4_spd=df[df['Parameter']== 'AW4010R']

# Print the dataframe
print(df.head())

print(sc_mntm_y.head())

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()
fig7 = plt.figure()
fig8 = plt.figure()

ax1 = fig1.add_subplot()
ax2 = fig2.add_subplot()
ax3 = fig3.add_subplot()
ax4 = fig4.add_subplot()
ax5 = fig5.add_subplot()
ax6 = fig6.add_subplot()
ax7 = fig7.add_subplot()
ax8 = fig8.add_subplot()


ax1.plot(sc_mntm_x['End time'], sc_mntm_x['Mean'], label='AW0004R')
ax2.plot(sc_mntm_y['End time'], sc_mntm_y['Mean'], label='AW0005R')
ax3.plot(sc_mntm_z['End time'], sc_mntm_z['Mean'], label='AW0006R')
ax4.plot(rw1_spd['End time'], rw1_spd['Mean'], label='AW1010R')
ax5.plot(rw2_spd['End time'], rw2_spd['Mean'], label='AW2010R')
ax6.plot(rw3_spd['End time'], rw3_spd['Mean'], label='AW3010R')
ax7.plot(rw4_spd['End time'], rw4_spd['Mean'], label='AW4010R')
ax8.plot(sc_mntm_tot['End time'], sc_mntm_tot['Mean'], label='HA0677D')

ax1.set_xlabel('End time')
ax1.set_ylabel('Mean')
ax1.set_title('SC x-axis ang-momentum')

ax2.set_xlabel('End time')
ax2.set_ylabel('Mean')
ax2.set_title('SC y-axis ang-momentum')

ax3.set_xlabel('End time')
ax3.set_ylabel('Mean')
ax3.set_title('SC z-axis ang-momentum')

ax4.set_xlabel('End time')
ax4.set_ylabel('Mean')
ax4.set_title('RW1 speed')

ax5.set_xlabel('End time')
ax5.set_ylabel('Mean')
ax5.set_title('RW2 speed')

ax6.set_xlabel('End time')
ax6.set_ylabel('Mean')
ax6.set_title('RW3 speed')

ax7.set_xlabel('End time')
ax7.set_ylabel('Mean')
ax7.set_title('RW4 speed')

ax8.set_xlabel('End time')
ax8.set_ylabel('Mean')
ax8.set_title('SC total ang-momentum')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()
ax8.legend()

plt.show()


#fig, ax = plt.subplots(nrows=8, ncols=1)
#plt.plot(df_AW0004R['End time'], df_AW0004R['Mean'], label='AW0004R')
#plt.plot(df_AW0005R['End time'], df_AW0005R['Mean'], label='AW0005R')
#plt.plot(df_AW0006R['End time'], df_AW0006R['Mean'], label='AW0006R')
#plt.plot(df_AW1010R['End time'], df_AW1010R['Mean'], label='AW1010R')
#plt.plot(df_AW2010R['End time'], df_AW2010R['Mean'], label='AW2010R')
#plt.plot(df_AW3010R['End time'], df_AW3010R['Mean'], label='AW3010R')
#plt.plot(df_AW4010R['End time'], df_AW4010R['Mean'], label='AW4010R')
#plt.plot(df_HA0677D['End time'], df_HA0677D['Mean'], label='HA0677D')











