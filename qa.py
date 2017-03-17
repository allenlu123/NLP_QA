import nltk
import string
from bllipparser import RerankingParser

# download and install a basic unified parsing model (Wall Street Journal)
# sudo python -m nltk.downloader bllip_wsj_no_aux
model_dir = nltk.data.find('models/bllip_wsj_no_aux').path
rrp = RerankingParser.from_unified_model_dir(model_dir)

def getTags(sentence):
	return nltk.pos_tag(nltk.word_tokenize(sentence))

def getParseTree(sentence):
	return rrp.simple_parse(sentence)

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
			