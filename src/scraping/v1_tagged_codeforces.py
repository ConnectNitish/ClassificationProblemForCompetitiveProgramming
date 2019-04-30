import requests
import csv
import copy
from bs4 import BeautifulSoup
import sys

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
f = open("v2_out.txt", "a")

def log_file(x="",y="",z="",a=""):
    s = str(x)+str(y)+str(z)+str(a)
    f.write(s)
    f.write("\n")


def is_page_loaded(result,function_name):
    if(result.status_code!=200):
        log_file("page could not be loaded -",function_name)
        return True
    return False

def get_code(sol_link):
    result = requests.get("https://codeforces.com/"+sol_link , headers=headers ,verify=False)
    if(is_page_loaded(result,"get_code")):
        return None
    code_src = result.content
    code_soup = BeautifulSoup(code_src,'lxml')
    table = code_soup.find("pre")
    if(table==None):
        log_file("no pre tag found in html..hence no solution was found")
        return None
    return table.text
        

        
def get_accepted_sols(param_link,lang_req):
    total_no_of_pages = 0
    result = requests.get("https://codeforces.com"+param_link+"/page/0" , headers=headers ,verify=False)
    if(is_page_loaded(result,"get_accepted_sols")):
        return None
    page2_src = result.content
    bs = BeautifulSoup(page2_src,'lxml')
    all_div = bs.findAll("div", {"class": "pagination"})
    if(all_div==None):
        log_file("no solutions page was found with link ","https://codeforces.com"+param_link+"/page/0")
        return None

    mylist = []
    for li in all_div:
            mylist.append(li)
            
    p_no = None
    for elem in mylist:
            if(elem):
                    p_no = elem.find_all("span",{"class":"page-index"})
    if(p_no == None):
        log_file("no span class:page-index found with link ","https://codeforces.com"+param_link+"/page/0")
        return None
    for elem in p_no:
            total_no_of_pages = int(elem.text)
            
    
    
    ret_obj = {}
    ret_obj["sol"] = "no code found"
    ret_obj["author"] = " NA "
    ret_obj["time_taken"] = " NA "
    page_number_of_sol = 1
    flag = True
    c_flag = 0
    while flag:
        if(page_number_of_sol>total_no_of_pages):
            flag = False
            break
        result = requests.get("https://codeforces.com"+param_link+"/page/"+str(page_number_of_sol) , headers=headers ,verify=False)
        if(is_page_loaded(result,"get_accepted_sols with link "+"https://codeforces.com"+param_link+"/page/"+str(page_number_of_sol))):
            return None
        log_file("url = ","https://codeforces.com"+param_link+"/page/"+str(page_number_of_sol))
        page_number_of_sol = page_number_of_sol+1
        page2_src = result.content
        bs = BeautifulSoup(page2_src,'lxml')
        rows = bs.find_all('tr')
        if(rows==None):
            log_file("there are no rows in table for link ","https://codeforces.com"+param_link+"/page/"+str(page_number_of_sol))
            return None
        no_of_sols_for_a_question = 0
        for row in rows:
            tlist = []
            cols = row.find_all('td')
            for c in cols:
                tlist.append(c)

            if(len(tlist)==8):
                sol_lang  = tlist[4].text.strip()
                links = tlist[0].find("a")
                if(links == None):
                    log_file("no links are there in 1st row i.e solution link (solution id ) missing")
                    return None
                sol_link = links.attrs['href']
                no_of_sols_for_a_question = no_of_sols_for_a_question + 1
                # submission_ts = tlist[1].text.strip()
                # log_file("submission ts = ",submission_ts)
                sol_author = tlist[2].text.strip()
                # log_file("sol_author = ",sol_author)
                # problem_id  = tlist[3].text.strip()
                # sol_verdict = tlist[5].text.strip() 
                sol_time_taken = tlist[6].text.strip()
                # sol_mem_taken = tlist[7].text.strip()
                # log_file(problem_id)
                # log_file(sol_lang)
                # log_file(sol_verdict)
                # log_file(sol_time_taken)
                # log_file(sol_mem_taken)
                if(lang_req=="python" and sol_lang.find("ython")!=-1):
                    code = get_code(sol_link)
                    c_flag = c_flag + 1
                    ret_obj["sol"] = code
                    ret_obj["author"] = sol_author
                    ret_obj["time_taken"] = sol_time_taken
                elif(lang_req=="c++" and sol_lang.find("++")!=-1):
                    code = get_code(sol_link)
                    c_flag = c_flag + 1
                    ret_obj["sol"] = code
                    ret_obj["author"] = sol_author
                    ret_obj["time_taken"] = sol_time_taken
                if(c_flag==1):
                    return ret_obj
            else:
                pass
                # log_file("no columns in table")
        if(no_of_sols_for_a_question<50):
            flag = False
            break
    # log_file("no of pages = ",page_number_of_sol)
    return ret_obj





def get_problem_statement(param):
    
    param = param.find('a')
    if(param == None):
        log_file("no <a> found in get_problem_statement()")
        return None
    p_link = param.attrs['href']
    result = requests.get("https://codeforces.com"+str(p_link) , headers=headers ,verify=False)
    if(is_page_loaded(result,"get_problem_statement()")):
        return None
    code_src = result.content
    code_soup = BeautifulSoup(code_src,'lxml')
    
    p_div = code_soup.find("div", {"class": "problem-statement"})
    text = ""
    elem_in_p_div = []
    if(p_div==None):
        return None
    for elem in p_div:
        elem_in_p_div.append(elem)
    p_tags = elem_in_p_div[1].find_all("p")
    for para in p_tags:
        text += para.text
    # log_file(text)
    return text

def get_problem_tag(param):
    problem_name = ""
    tags = []
    all_href = param.find_all("a")
    if(all_href == None):
        log_file("no <a> to get problem tag in get_problem_tag()")
        return None
    for a_href in all_href:
        temp = str(a_href.attrs['href'])
        if(temp.find("/problemset/problem")!=-1):
            problem_name = a_href.text
            problem_name = problem_name.strip()
            # log_file()
        if(temp.find("/problemset?tags=")!=-1):
            tags.append(temp.split("=")[1])
    final_tags = []
    for t in tags:
        final_tags.append(t.replace("+"," "))
    problem_tags = ""
    for t in final_tags:
        problem_tags = problem_tags + str(t) + ","
    problem_tags = problem_tags[:len(problem_tags)-1] 
    # log_file(final_tags)
    return problem_tags,problem_name
    


def get_details(input_page_no):
    global input_p_id
    missed_problem_ids = []
    captured_problem_ids = []
    start = False
    no_of_untagged = 0
    total_no_of_qus = 0
    page_no = input_page_no
    flag = True
    while flag:
        # flag = False
        page_no = page_no+1
        print("on page ",page_no)
        try:
            log_file("link " ,          "https://codeforces.com/problemset/"+"page/"+str(page_no))
            result = requests.get("https://codeforces.com/problemset/"+"page/"+str(page_no) , headers=headers, verify=False)
        except:
            print("exception in start of while :page_no ",page_no)
            return page_no,""
        if(is_page_loaded(result,"get_details() ")):
            print("page couldnt be loaded ",page_no)
            return page_no,""
        
        if(page_no==51):
                break
        
        src = result.content
        soup = BeautifulSoup(src,'lxml')
        rows = soup.find_all('tr')
        if(rows == None):
            log_file(" no data in page: ","https://codeforces.com/problemset/"+"page/"+str(page_no))
            exit()
        for row in rows:
            tlist = []
            cols=row.find_all('td')
            if(cols == None):
                log_file("no <td> in <tr> for row = ",row)
                continue
    
            for c in cols:
                    tlist.append(c)
            difficulty_value = ""
            problem_tag = ""
            problem_name = ""
            problem_statement = ""
            problem_id = ""
            try:
                if(len(tlist)==5):
                        total_no_of_qus = total_no_of_qus + 1
                        difficulty_value = tlist[3].text.strip()
                        problem_tag,problem_name = get_problem_tag(tlist[1])
                if(len(tlist)==5 and problem_tag!=""):
                    csv_row = []
                    problem_id = tlist[0].text.strip()
                    # log_file("problem_id = ",problem_id)
                    problem_statement = get_problem_statement(tlist[1])
                    # log_file("problem_name ",problem_name)
                    # log_file("problem_tag ",problem_tag)
                    
                    links = tlist[4].find("a")
                    if(links == None):
                        log_file("no links for solution for question id",problem_id)
                        missed_problem_ids.append(problem_id)
                        log_file("missed problem id ",problem_id)
                        continue

                    sol_link = links.attrs['href']
                    

                    if(input_p_id==problem_id):
                        start = True
                    elif(input_p_id==""):
                        start = True
                    if(start):
                        obj_returned = get_accepted_sols(sol_link,"c++")
                        # to do error handling
                        if(problem_id==""):
                            missed_problem_ids.append("id missing")
                            log_file("missed problem id ",problem_id)
                            continue
                        if(problem_name=="" or problem_name==None):
                            missed_problem_ids.append(problem_id)
                            log_file("missed problem id ",problem_id)
                            continue
                        if(problem_statement==None or problem_statement==""):
                            missed_problem_ids.append(problem_id)
                            log_file("missed problem id ",problem_id)
                            continue
                        if(problem_tag==None or problem_tag==""):
                            missed_problem_ids.append(problem_id)
                            log_file("missed problem id ",problem_id)
                            continue
                        if(difficulty_value=="" or difficulty_value==None):
                            difficulty_value = ""
                        if(obj_returned == None):
                            missed_problem_ids.append(problem_id)
                            log_file("missed problem id ",problem_id)
                            continue
                        if(obj_returned["sol"]==None or obj_returned["sol"]==""):
                            missed_problem_ids.append(problem_id)
                            continue
                        captured_problem_ids.append(problem_id)
                        log_file("captured id ",problem_id)
                        csv_row.append(problem_id)
                        csv_row.append(problem_name)
                        csv_row.append(problem_statement)
                        csv_row.append(problem_tag)
                        csv_row.append(difficulty_value)
                        csv_row.append(obj_returned["sol"])
                        csv_row.append(obj_returned["time_taken"])
                        csv_row.append(obj_returned["author"])
                        

                        if(problem_statement!=None):
                            with open('v2_tagged_codeforces.csv', 'a') as csvFile:
                                    writer = csv.writer(csvFile)

                                    writer.writerow(csv_row)
                            csvFile.close()
                    
                elif(problem_tag=="" and len(tlist)==5):
                    pass
                    # no_of_untagged = no_of_untagged + 1
                    # csv_row = []
                    # problem_id = tlist[0].text.strip()
                    # log_file("problem_id = ",problem_id)
                    # problem_statement = get_problem_statement(tlist[1])
                    # links = tlist[4].find("a")
                    # if(links==None):
                    #         continue
                    # sol_link = links.attrs['href']
                    # log_file("1")
                    # obj_returned = get_accepted_sols(sol_link,str(input_prg))
                    # log_file("2")
                    # log_file(input_p_id,problem_id)
                    # if(input_p_id==problem_id):
                    #         start = True
                    # # start = True
                    # if(start):
                    #         csv_row.append(problem_id)
                    #         csv_row.append(problem_name)
                    #         csv_row.append(problem_statement)
                    #         csv_row.append(obj_returned["sol"])
                    #         csv_row.append(obj_returned["time_taken"])
                    #         csv_row.append(obj_returned["author"])
                    #         with open('untagged_codeforces.csv', 'a') as csvFile:
                    #                 writer = csv.writer(csvFile)
            
                    #                 writer.writerow(csv_row)
                    #         csvFile.close()

                
                    log_file("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            except:
                print("exception at end of while page_no ",page_no," pid ",problem_id)
                return page_no,problem_id
        if(total_no_of_qus<100):
                flag = False
                break
    log_file("total no of pages =============",page_no)
    log_file("number of total_no_of_qus  =====",total_no_of_qus)
    log_file("number of missed ids ",len(missed_problem_ids))
    log_file("number of captured ids ",len(captured_problem_ids))


input_prg = input("enter c++ or python ")
input_page_no = input("enter page no ")
input_page_no = int(input_page_no)
input_p_id  = input("enter problem id ")
if(input_page_no==1):
    csv_row = ["id","name","problem statement","tags","difficulty","solution","time_taken","author"]
    with open('v2_tagged_codeforces.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(csv_row)
    csvFile.close()

input_page_no,input_p_id = get_details(input_page_no)
print("input_page_no ",input_page_no," input_p_id ",input_p_id)
while(input_page_no<51):
    input_page_no = input_page_no - 1
    input_page_no,input_p_id = get_details(input_page_no)
f.close()