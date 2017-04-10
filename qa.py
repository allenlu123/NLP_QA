import nltk
import string
from bllipparser import RerankingParser
import re

#Python 3
# download and install a basic unified parsing model (Wall Street Journal)
# sudo python -m nltk.downloader bllip_wsj_no_aux
model_dir = nltk.data.find('models/bllip_wsj_no_aux').path
rrp = RerankingParser.from_unified_model_dir(model_dir)

def textToSentences(text):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	return tokenizer.tokenize(text)

def getTags(sentence):
	return nltk.pos_tag(nltk.word_tokenize(sentence))

def scoredParseToTree(scored_parse):
	return nltk.tree.Tree.fromstring(str(scored_parse.ptb_parse))

def getParseTree(sentence):
	words = []
	tag_map = {}
	for i,(word,tag) in enumerate(getTags(sentence)):
		words.append(word)
		if tag != None:
			tag_map[i] = tag

	nbest_list = rrp.parse_tagged(words,tag_map)

	for scored_parse in nbest_list:
		#return best parse tree
		return re.sub('\s+', ' ',str(scoredParseToTree(scored_parse))).strip()

class Tree(object):
	def __init__(self):
		self.child_tags = []
		self.children = []
		self.sentence = ''
		self.tag = ''

def makeTreeHelp(parsed,parsed_len,start,tag):
	curr_tree = Tree()
	curr_tree.tag = tag
	leaf = True
	i = start
	while i < parsed_len:
		if parsed[i] == '(':
			leaf = False
			space_ind = parsed.find(' ',i)
			new_tag = parsed[i + 1:space_ind]
			new_start = space_ind + 1
			(child,i) = makeTreeHelp(parsed,parsed_len,new_start,new_tag)
			#child sentence should not be empty
			if not child.sentence[0] in string.punctuation:
				curr_tree.sentence += ' ' + child.sentence
				curr_tree.sentence = curr_tree.sentence.strip()
			else:
				curr_tree.sentence += child.sentence
			curr_tree.child_tags.append((child.tag,child.sentence))
			curr_tree.children.append(child)
		elif parsed[i] == ')':
			if leaf:
				curr_tree.sentence = parsed[start:i]
			return (curr_tree,i + 1)
		else:
			i += 1

	return (curr_tree,i + 1)

def makeTree(parsed):
	parsed = parsed[parsed.index(' ') + 1:][:-1]

	(tree,i) = makeTreeHelp(parsed,len(parsed),0,'S1')
	if len(tree.children) == 1:
		return tree.children[0]
	return tree

def treeFromSentence(sentence):
	parsed = getParseTree(sentence)
	return makeTree(parsed)

def generateQuestions(text):
	sentences = textToSentences(text)
	questions = []
	for sentence in sentences:
		parts = sentence.split(',')
		for part in parts:
			t = treeFromSentence(part)
			parsed = t.child_tags
			if len(parsed) > 1:
				first = parsed[0]
				second = parsed[1]
				if first[0] == 'NP' and second[0] == 'VP':
					question = questionNPVP(first[1],second[1])
					if question != None:
						questions.append(question)
					else:
						parsed_vp = t.children[1].child_tags
						if len(parsed_vp) > 1:
							vp_first = parsed_vp[0]
							vp_second = parsed_vp[1]
							if vp_first[0].startswith('VB') and vp_second[0] == 'NP':
								questions.append('What ' + vp_first[1] + ' ' +
												vp_second[1] + '?')

			elif len(parsed) > 2:
				first = parsed[0]
				second = parsed[1]
				third = parsed[2]
				if second[0].startswith('VB') and third[0] == 'NP':
					questions.append('What ' + second[1] + ' ' + third[1] + '?')
	return questions

def isPerson(np):
	np_tags = getTags(np)
	if np_tags[0][0] == 'PRP':
		return True
	else:
		entities = nltk.chunk.ne_chunk(np_tags)
		for entity in entities:
			if isinstance(entity,nltk.tree.Tree) and entity._label == 'PERSON':
				return True
	return False	

def questionNPVP(np,vp):
	if isPerson(np):
		return 'Who ' + vp + '?'
	elif np != 'It':
		return 'What ' + vp + '?'
	return None

'''
def writeRuleHelp(preterminal,t,rules):
	if len(t.children) == 0:
		rhs = ' -> ' + t.sentence
		if not t.tag in rules:
			rules[t.tag] = {rhs:1}
		elif not rhs in rules[t.tag]:
			rules[t.tag][rhs] = 1
		else:
			rules[t.tag][rhs] += 1
		return
	rhs = ' -> ' + ' '.join([tag[0] for tag in t.child_tags])
	if not preterminal in rules:
		rules[preterminal] = {rhs:1}
	elif not rhs in rules[preterminal]:
		rules[preterminal][rhs] = 1
	else:
		rules[preterminal][rhs] += 1
	for child in t.children:
		writeRuleHelp(child.tag,child,rules)

def writeRules(file_in,file_out):
	f = open(file_in)
	arr = f.readlines()
	f.close()

	rules = {'ROOT': {}}
	for i in range(len(arr)):
		rules['ROOT'][' -> S1'] = 1
		writeRuleHelp('S1',treeFromSentence(arr[i]),rules)
	f = open(file_out,'w+')
	for key in rules.keys():
		for rhs in rules[key].keys():
			count = rules[key][rhs]
			f.write(key + rhs + ' ' + str(count) + '\n')
	f.close()
'''
			