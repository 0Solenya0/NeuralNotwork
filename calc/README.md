Training a transformer model to solve simple arithmetic expressions (addition, subtraction, multiplication, division)

The following is my thoughts about each version.

# v1
A standard transformer with positional embedding.
It's only aim is to solve addition.

I've included some samples at the end of the notebook to show some weaknesses.
It seems like the model just memorized the answers, and it doesn't generalize
to more than 3-digit numbers + single digit addition is also underrepresented
because of how the data was generated. Here are my next improvement ideas:
+ I think it might get better if the positions are fed to the model in a relative manner instead of doing embedding for each position.