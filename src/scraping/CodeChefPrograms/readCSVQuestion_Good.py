# importing csv module
import csv

# csv file name
filename = "third.csv"

#filenameList = ["first.csv","second.csv","third.csv"]

filenameList = ["questions.csv"]

data_of_solution_question = []

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
		#QCode,Title,link,level,statement,Author,Tester,Editorial,Tags,Date Added,Time Limit,Source Limit,Languages
		data_of_solution_question.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12]])
	
	print(" COmpleted " + filename)



filenameList = ["first.csv","second.csv","third.csv"]

data_of_solution = {}

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
		data_of_solution.update({row[0]:row[1]})
	
	print(" COmpleted " + filename)



filenameList = ["solutions.csv"]

data_of_solution_details = {}

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
		#QCode,SolutionID,timeago,UserID,Status,TimeTaken,MemTaken,Language,SolutionUrl
		if row[4] == 'accepted' and row[0] not in data_of_solution_details.keys():
			data_of_solution_details.update({row[0]:{row[1]:[row[2],row[3],row[4],row[5],row[6],row[7],row[8]]}})
	
	print(" COmpleted " + filename)

question_submission_id_key_list = data_of_solution.keys()
question_code_accepted_key_list = data_of_solution_details.keys()


print(len(data_of_solution_question))
print(len(data_of_solution))
print(len(data_of_solution_details))

#QCode,Title,link,level,statement,Author,Tester,Editorial,Tags,Date Added,Time Limit,Source Limit,Languages
#data_of_solution_question.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12]])

#QCode,SolutionID,timeago,UserID,Status,TimeTaken,MemTaken,Language,SolutionUrl
#data_of_solution_details.update({row[0]:{row[1]:[row[2],row[3],row[4],row[5],row[6],row[7],row[8]]}})

#fields = ['QCode','Title','link','level','statement','Author','Tester','Editorial','Tags','Date Added','Time Limit','Source Limit','Languages']

fields = ['QuestionCode','Title','Questionlink','Difficultylevel','Problem Statement','Editorial','Tags','Time Limit','Languages','Solution']

#[temp_question_code,temp_Title,temp_question_link,temp_Level,temp_problem_statement,temp_Editorial,temp_Tags,temp_time_taken,temp_language,temp_actualSolutionText]

temp_actualSolutionText = ""
temp_question_code=""
temp_time_taken=""
temp_language=""
temp_Level=""
temp_problem_statement=""
temp_Editorial=""
temp_Tags=""
temp_question_link=""
temp_Title=""

final_result_to_put_in_file=[]

for item in data_of_solution_question:
	question_code = item[0]
	found_temp_value = 0
	if question_code in question_code_accepted_key_list:
		value_of_accepted_solution = data_of_solution_details[question_code]
		#timeago,UserID,Status,TimeTaken,MemTaken,Language,SolutionUrl
		submission_id_of_accepted_solution_list = value_of_accepted_solution.keys()
		submission_id_of_accepted_solution_list_values = value_of_accepted_solution.values()

		#print("--------------------------------------------------------")
		#print(submission_id_of_accepted_solution_list_values)
		

		if len(submission_id_of_accepted_solution_list)>0:
			submission_id_of_accepted_solution = submission_id_of_accepted_solution_list[0]
			if submission_id_of_accepted_solution in question_submission_id_key_list:
				temp_actualSolutionText = data_of_solution[submission_id_of_accepted_solution]

				found_temp_value = 1

				temp_question_code = question_code
				temp_time_taken = submission_id_of_accepted_solution_list_values[0][3]
				temp_language = submission_id_of_accepted_solution_list_values[0][5]
				temp_Level = item[3]
				temp_problem_statement = item[4]
				temp_Editorial = item[7]
				temp_Tags = item[8]
				temp_question_link = item[2]
				temp_Title = item[1]

				final_result_to_put_in_file.append([temp_question_code,temp_Title,temp_question_link,temp_Level,temp_problem_statement,temp_Editorial,temp_Tags,temp_time_taken,temp_language,temp_actualSolutionText])



print("TOTAL WRITE")
print(len(final_result_to_put_in_file))
#print((data_of_solution_question[0]))
#print((data_of_solution[0]))
#print((data_of_solution_details['A1']))


#print(data_of_solution_question)


#fields = ['SolutionID', 'Solution'] 
#QCode,Title,link,level,statement,Author,Tester,Editorial,Tags,Date Added,Time Limit,Source Limit,Languages
#fields = ['QCode','Title','link','level','statement','Author','Tester','Editorial','Tags','Date Added','Time Limit','Source Limit','Languages']

# name of csv file 
filename = "AllDetails.csv"
  
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(final_result_to_put_in_file)

print("writing completed")
'''
#csvFile.close()
'''
