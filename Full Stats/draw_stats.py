import matplotlib.pyplot as plt
import numpy as np
import ast

def draw_plot(fN_list,interval,labels):
	t=0
	for fN in fN_list:
		fp = open(fN,'r')
		big_lst=[]
		for line in fp:
			line=line.replace('\n','')
			my_tup=ast.literal_eval(line)
			big_lst.append(my_tup)
	
		mean_lst=[]

		score_col=np.array([i[4] for i in big_lst])
		print fN
		print "Max score: "+str(np.max(score_col))
		print "90th_percentile: "+str(np.percentile(score_col, 90))
		print "Mean score: "+str(np.mean(score_col))
		print "Median score: "+str(np.median(score_col))
		print "10th_percentile: "+str(np.percentile(score_col,10))
		print "Min score: "+str(np.min(score_col))

		i=0
		while (i<len(big_lst)):
			j=0
			temp_array=[]
			while(j<interval):
				try:
					temp_array.append(big_lst[i][4])
				except:
					print (len(big_lst),i)
					print big_lst[i]
					return
				i+=1
				j+=1
			mean_lst.append(np.mean(temp_array))

		y = mean_lst
		print str(len(mean_lst))+" points plotted out of "+str(len(big_lst))
		print
		x = [i for i in range(len(y))]
		plt.plot(x,y,label=labels[t])

		t+=1
	
	plt.xlabel("Game count... (Averaged every "+str(interval)+" )")
	plt.ylabel("Score...")
	plt.legend(loc='best')
	plt.title("Compare Fine_tune, Pre_train and from Scratch on deadly corridor")
	plt.show()


fileName1 = 'Phase4/new_ddqn_per_corridor_more_actions_new_center_fine_tune_ph4.txt'
fileName2 = 'Phase4/new_ddqn_per_corridor_more_actions_new_center_pre_train_ph4.txt'
fileName3 = 'Phase4/new_ddqn_per_corridor_ph4.txt'
labels = ['Fine_tune','Pre_train','Scratch']
fileList=[fileName1,fileName2,fileName3]
#fileList = [fileName1]
draw_plot(fileList, interval=20,labels=labels)	
