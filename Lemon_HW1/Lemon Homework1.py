# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:25:18 2019

@author: Lemon Lin Reimer
"""
#%% Example Code:

for word in range(0, 100):

    print (word)

#%% 1. Change the example code to count up in increments of 2

for word in range(0, 100, 2):
    print(word)

#%% 2. Change the example to count up in increments of 0.5

for word in range(0, 100, 0.5):
    print(word)

#%% 3. Change the loop statement below to an inline for loop statement:

the_sentence = 'the student went above and beyond the call of duty on the homework'
my_ar = list()

for word in the_sentence.split():

    my_ar.append(word)

inlineloop = [(word) for word in the_sentence.split()]
print(inlineloop)

#%% 4. Change the below code to NOT include the word orange:

the_sentence = 'the orange cat jumped over the dog, but the orange cat landed on another orange cat'

my_ar = [word for word in the_sentence.split() if word != "orange"]
print(my_ar)

#%% 5. Create a program that removes all special characters from the sentence:

the_sentence = 'woah!!! @student really #impressed me, & so did you!!!'

import re

clean_sentence = re.sub('[^A-z0-9]+', " ", the_sentence)
print(clean_sentence)

#%% 6. Write a program that creates a sentence, each word separated by only one 
# space, out of the following array and replace any special characters with no 
# space, except the '!' characters

the_ar = ['woah!!!','the','@student',' really^','# impressed','me,','and&','so','did','??you!!!']

sentence = " " .join(the_ar)

new_ar = re.sub('[^A-Za-z!]+', " ", sentence)
print(new_ar)

#%% 7. Create a program that iterates 1 through 10 and returns 'the number: <number> is even' 
# if the number is even and 'the number: <number> is odd’ if the number is odd.
 
def evenodd(*args):
    tmp = iter(args)
    for val in tmp:
        if (val % 2) == 0:
            print("the number: " + str(val) + " is even")
        else:
            print("the number: " + str(val) + " is odd")
    return(tmp)
    
print(evenodd(1,2,3,4,5,6,7,8,9,10))

#%% 8. Create a program that counts the occurrence of each word in an arbitrary 
# sentence (variable called the_sentence) and stores each unique 'word' in a 
# dictionary where the key is the word and the value is the number of occurrences of that word.

def occur(the_sentence):
    tmp = re.split(" ", the_sentence)
    tmp_dict = {}
    for word in tmp:
        tmp_dict.update({str(word): tmp.count(word)})
    return(tmp_dict)

test = occur("this is my test sentence which contains the word sentence twice")
print(test)

#%% 9. Create a program that replaces any word of a sentence (variable called the_sentence) with an arbitrary word

def replace(the_sentence, word, newword):
    tmp = the_sentence.replace(word, newword)
    return(tmp)

sent1 = ("this is a sentence with only one word replaced: cat")
sent2 = ("this is a sentence with two words replaced: cat and cat")

test1 = replace(sent1, "cat", "dog")
print(test1)

test2 = replace(sent2, "cat", "dog")
print(test2)

# bonus: I wanted to build in an error for when the word is not present in the_sentence

def replace2(the_sentence, word, newword):
    tmp = the_sentence.replace(word, newword)
    err = re.split(" ", the_sentence)
    if err != word:
        print("Error: the word " + word + " is not in this sentence!")
    else:
        return(tmp)

sent3 = ("this is a sentence that doesn't refer to the best pets by name")

replace2(sent3, "cat", "dog")

#%% 10. Cleanse the following sentence by removing all special characters except 
# when the hyphen (-) joins to two words and exclamation (!) points
 
the_sentence = 'The impact*of data-driven$^%&marketing approaches!!'

new_sentence = re.sub('[^A-Za-z!-]+', " ", the_sentence)
print(new_sentence)

#%% 11. Write a python program that accepts an arbitrary sentence and returns a 
# dictionary that has each unique 'character' as a key and count of that character as the value.
 
def lettercount(the_sentence):
    tmp_dict = {}
    for word in the_sentence:
        tmp_dict.update({str(word): the_sentence.count(word)})
    return(tmp_dict)

sent4 = ("this sentence sure has a lot of e in it eeeeeeeeeeeeeeeee!! AND ITS CASE SENSITIVE")
test4 = lettercount(sent4)
print(test4)