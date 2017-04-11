'''
To run answer questions, provide questions file and text file
python3 run.py 0 questions_file text_file

To generate questions, provide text file and number of questions
python3 run.py 1 text_file n
'''

import qa
import sys

if __name__ == '__main__':
	job = int(sys.argv[1])
	if job == 0:
		questions_file = sys.argv[2]
		text_file = sys.argv[3]
		qa.answerQuestions(questions_file,text_file)
	else:
		text_file = sys.argv[2]
		n = int(sys.argv[3])
		qa.questionsFromText(text_file,n)