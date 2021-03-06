#FIRST RUN Select_LCBG_BESSEL.py to calculate absolute magnitudes and redshifts for objects
plt.figure(figsize=(8,8))
plt.plot(zbest,M,',',color='black')
plt.xlim(0,1)
#plt.grid()
plt.xlabel('Redshift',fontsize=16)
plt.ylabel('M$_{B}$ (mag)',fontsize=16)
plt.fill_between(np.array([0.01,0.4]),-23.5,-22.5,color='0.25',alpha=0.1)
plt.fill_between(np.array([0.2,0.4]),-16,-17,color='0.25',alpha=0.1)
plt.fill_between(np.array([0.4,0.6]),-17.5,-17,color='0.25',alpha=0.1)
plt.fill_between(np.array([0.6,0.8]),-19.5,-19,color='0.25',alpha=0.1)
plt.fill_between(np.array([0.8,1]),-21.5,-21,color='0.25',alpha=0.1)
plt.subplots_adjust(right=0.98,left=0.11,top=0.98,bottom=0.07)
plt.savefig('/mnt/c/Users/lrh03/Documents/Work/THESIS/ThesisTalk/REDSHIFT_ABSMAG.pdf')
