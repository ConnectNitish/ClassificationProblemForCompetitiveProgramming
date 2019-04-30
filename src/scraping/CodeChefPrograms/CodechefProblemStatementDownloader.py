
# Script to help download problem statements from codechef

import urllib.request, bs4, re, os, time, sys
import requests
from bs4 import BeautifulSoup
import os

# Function showing the progress of the download count
def progress(count = ''):
    sys.stdout.write('%s\r' % (count))
    sys.stdout.flush()

problems = ['medium','school',  'easy', 'hard', 'challenge', 'extcontest']
programminglanguagesuffix = ['py','cpp','c','java','js']
algorithm_tag_list = ['maths','gcd','fermat','euler','bfs','dfs']
data_structure_tag_list = ['graph','array','dp','segmenttree','tree','bit','stack','queue','linklist']


data_value_problem_code = []



# {
#     'Code':'CHNMD',
#     'ProblemStemetent':'Question',
#     'Tags':'AllTags',
#     'Solution':'',
#     'Type':'',
#     'EditorialLink':''
#     'Language':''
# }

# <tr>
# <td width="14%">Editorial</td>
# <td><a href="https://discuss.codechef.com/problems/CHNUM">https://discuss.codechef.com/problems/CHNUM</a></td>
# </tr>

for idx, problem in enumerate(problems):
    

    # if idx >= 1:
    #     os.chdir('..')

    # Create a new directory
    # os.mkdir(problem)
    # If directory exists, go to that directory to save all the files
    # if os.path.exists(problem):
    # os.chdir(problem)
    # web address of codechef website
    codechefWebsite = 'https://www.codechef.com'
    print(codechefWebsite + '/problems/'+problem) 
    
    # Get the HTML from teh website
    # getHTML = urllib.request.urlopen(codechefWebsite + '/problems/' + problem)
    # # Read the data
    # data = getHTML.read()

    source_code = requests.get(codechefWebsite + '/problems/' + problem)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, "lxml")

    # # Parse the HTML data
    # soup = bs4.BeautifulSoup(data, 'html.parser')
    # # Find the content-wrapper for all probl statements
    check = soup.find_all(class_ = 'content-wrapper')

    # Find specific href tags that have 'problems/' in them
    result = soup.find_all(href = re.compile('problems/'))

    # print(result)
    # pass

    
    data_value = {}

    downloaded = 0

    for i in range(15, len(result)):

        ProblemCode = "" # Done
        ProblemStatement = ""
        Tags = ""
        Solution = ""
        DifficultyLevel = problem # Done
        EditorialLink = "" # Done
        Language = ""

        settersolution = ""
        editorialsolution = ""
        testersolution = ""
        settersolutionlink = ""
        editorialsolutionlink = ""
        testersolutionlink = ""

        checkResult = result[i]['href']
        ProblemCode = checkResult

        print(checkResult)
        
        try:
            


            source_code = requests.get(codechefWebsite + checkResult)
            plain_text = source_code.text
            soup = BeautifulSoup(plain_text, "lxml")

            gettingpro_list = soup.find_all(['div'])

            for item in gettingpro_list:
                if item!=None and (re.search("problems statements".lower().replace(" ",""),str(item.text).replace(" ","").lower())
                or re.search("problem statements".lower().replace(" ",""),str(item.text).replace(" ","").lower())):
                    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    # print(item.text)
                    entire_input_text = item.text
                    # print(item.text.index("as well."))
                    ProblemStatement = entire_input_text




            geteditorialLink = soup.find_all(['tr'])

            finaleditorialLink = ""

            # print(geteditorialLink)

            

            for item in geteditorialLink:
                # print(str(item))
                # print()
                if item!=None and re.search('Editorial',str(item)):
                    # print('AAA--------------')
                    # print(item.find_all(['a']))
                    anchor_data = item.find_all(['a'])
                    if len(anchor_data)==1 and re.search('discuss.codechef',str(item.find_all(['a']))):
                        # print('GO TO EDITORIAL')
                        # print(item.find_all(['a'])[0]['href'])
                        finaleditorialLink = item.find_all(['a'])[0]['href']
                elif item!=None and re.search('<td width="14%">Tags</td>'.lower(),str(item).lower()):
                    # print("---------------TAGS----------------------------")
                    # print(item)
                    anchor_data = item.find_all(['a'])
                    # print(anchor_data)
                    for local_item in anchor_data:
                        try:
                            # print(local_item.text)
                            if local_item.text != "":
                                Tags = Tags + local_item.text + ","
                        except:
                            print("----------EXCEPTION IN GETTING TAG")
            
            # print("---------------TAGS----------------------------")
            # print(Tags)

            anchortag_links = []
            pre_data_links = []

            if finaleditorialLink != "" :

                # print("GOTO EDITORIAL")
                # print(finaleditorialLink)

                EditorialLink = finaleditorialLink

                source_code = requests.get(finaleditorialLink)
                plain_text = source_code.text
                soup = BeautifulSoup(plain_text, "lxml")

                # html_text = open(finaleditorialLink).read()
                # soup_data = BeautifulSoup(html_text)
                
                # allpre = soup_data.find_all(['pre'])
                # print(allpre)

                # solution wrap up in pre tag 
                # print("PRE TAG")
                allpossiblesolutiondiv = soup.find_all(['pre'])
                for item in allpossiblesolutiondiv:
                    # print("PRE TAG")
                    # print(item)
                    if item != None:
                        item = str(item)
                        pre_data_links.append(item.replace("<pre>","").replace("</pre>",""))

                # print("INSIDE a tag")
                allpossiblesolutiondiv = soup.find_all(['a'])
                pair = {}
                for item in allpossiblesolutiondiv:
                    if str(item)!="":
                        if item!=None and re.search('author',str(item)) and re.search('solution',str(item)):
                            # print('author')
                            # print(item)
                            pair = {'author':item}
                        if item!=None and re.search('tester',str(item)) and re.search('solution',str(item)):
                            # print('tester')
                            # print(item)
                            pair = {'tester':item}
                        if item!=None and re.search('editorial',str(item)) and re.search('solution',str(item)):
                            # print('editorial')
                            # print(item)
                            pair = {'editorial':item}
                    # print()
                    if pair!={} and pair not in anchortag_links:
                        anchortag_links.append(pair)

                data_value[checkResult] = { 'anchortag' : anchortag_links , 'pretaglinks' : pre_data_links , 
                'EditorialLink' : EditorialLink , 'ProblemCode' : ProblemCode , 'Tags' : Tags ,
                'ProblemStatement':ProblemStatement };                         
                
                Tags = ""
                ProblemCode = ""
                EditorialLink = ""
                ProblemStatement = ""
                
                downloaded += 1

                if downloaded > 10:
                    break

        except urllib.error.HTTPError:
            # This exception has to bbe caught, else you might get an error saying Service temporarily unavailable
            print("----------------EXCEPTION CAUGHT-------------------")
            time.sleep(2)

    sys.stdout.write(']')
    # print(len(data_value))

    for item,values in data_value.items():

        ProblemCode = "" # Done
        ProblemStatement = ""
        Tags = ""
        Solution = ""
        DifficultyLevel = problem # Done
        EditorialLink = "" # Done
        Language = ""

        settersolution = ""
        editorialsolution = ""
        testersolution = ""
        settersolutionlink = ""
        editorialsolutionlink = ""
        testersolutionlink = ""

        for _item,_values in values.items():
            if len(_values) > 0:
                if _item == 'anchortag':
                    for __value in _values:
                        for a1,b1 in __value.items():                                
                            link_value_final = b1['href']
                            for item in programminglanguagesuffix:
                                if link_value_final.endswith("."+item): 
                                    if a1 == 'setter':
                                        settersolutionlink = b1['href']
                                    elif settersolution=="" and a1 == 'tester':
                                        testersolutionlink = b1['href']
                                    elif testersolutionlink=="" and a1 == 'editorial':
                                        editorialsolutionlink = b1['href']
                                    break;

                    if settersolutionlink !="":
                        for item in programminglanguagesuffix:
                            if settersolutionlink.endswith("."+item): 
                                source_codetemp = requests.get(settersolutionlink)
                                plain_text = source_codetemp.text
                                settersolution = plain_text
                                # print()
                                # print(ProblemCode)
                                # print("setter Solution")
                                # print(settersolution) 
                                
                    
                    if testersolutionlink !="":
                        for item in programminglanguagesuffix:
                            if testersolutionlink.endswith("."+item): 
                                source_codetemp = requests.get(testersolutionlink)
                                plain_text = source_codetemp.text
                                testersolution = plain_text
                                # print()
                                # print(ProblemCode)
                                # print("tester Solution")
                                # print(testersolution) 
                    
                    if editorialsolutionlink !="":
                        for item in programminglanguagesuffix:
                            if editorialsolutionlink.endswith("."+item): 
                                source_codetemp = requests.get(editorialsolutionlink)
                                plain_text = source_codetemp.text
                                editorialsolution = plain_text
                                # print()
                                # print(ProblemCode)
                                # print("editorial Solution")
                                # print(editorialsolution) 


                elif _item == 'pretaglinks':
                    # print(_values)
                    print("pretaglinks Solution Fetching ")
                    if settersolutionlink=="" and testersolutionlink=="" and editorialsolutionlink=="":
                        listofavaialablecode = _values
                        print(len(listofavaialablecode))
                        if len(listofavaialablecode) >= 3:
                            settersolution = listofavaialablecode[0]
                            testersolution = listofavaialablecode[1]
                            editorialsolution = listofavaialablecode[2]
                        elif len(listofavaialablecode) >= 2:
                            settersolution = listofavaialablecode[0]
                            testersolution = listofavaialablecode[1]
                        elif len(listofavaialablecode) == 1:
                            settersolution = listofavaialablecode[0]
                                            
            # print("---------------------")
                else:
                    if _item == 'ProblemCode':
                        ProblemCode = _values
                    elif _item == 'EditorialLink':
                        EditorialLink = _values
                    elif _item == 'Tags':
                        Tags = _values
                    elif _item == 'ProblemStatement':
                        ProblemStatement = _values

        # print()
        # print(ProblemCode)
        # print("settersolution",len(str(settersolution)))
        # print("testersolution",len(str(testersolution)))
        # print("editorialsolution",len(str(editorialsolution)))

        if settersolution!="" :
            Solution = settersolution
        elif testersolution!="":
            Solution = testersolution
        elif editorialsolution!="":
            Solution = editorialsolution

        final_key_Value_pair = {'Code':str(ProblemCode),
        'Solution':str(Solution).strip(),'Type':str(DifficultyLevel), 'EditorialLink':str(EditorialLink),'Language':str(Language)
        ,'Tags':Tags , 'ProblemStatement':ProblemStatement }

        # print(final_key_Value_pair)
        
        data_value_problem_code.append(final_key_Value_pair)

# print(data_value_problem_code)
import csv

keys = data_value_problem_code[0].keys()
with open('people.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data_value_problem_code)
