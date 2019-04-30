# importing csv module
import csv

# csv file name
filename = "third.csv"

filenameList = ["first.csv","second.csv","third.csv"]

data_of_solution = []

for item in filenameList:
	filename = item

	# initializing the titles and rows list
	fields = []
	rows = []

	# reading csv file
	with open(filename, 'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile)

		# extracting field names through first row
		fields = csvreader.next()

		# extracting each data row one by one
		for row in csvreader:
		        rows.append(row)

		# get total number of rows
		print("Total no. of rows: %d"%(csvreader.line_num))

	# printing the field names
	print('Field names are:' + ', '.join(field for field in fields))

	# printing first 5 rows
	print('\nFirst 5 rows are:\n')
	
	#for row in rows[:5]:
	for row in rows:
		# parsing each column of a row
		#for col in row:
		#        print("%10s"%col)
		#print('\n')
		data_of_solution.append([row[0],row[1]])
	
	print(" COmpleted " + filename)

fields = ['SolutionID', 'Solution'] 

# name of csv file 
filename = "SolutionID_Solution.csv"
  
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(data_of_solution)

print("writing completed")

#csvFile.close()

