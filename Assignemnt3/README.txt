Basic description and tutorial on how to use our code:

Folder "car" : includes the solution to the classification problem "car present/not present" based on the tutorial referenced in the task description.

Folder "fruit" : includes our solution to the fruit classification problem, contained in the file fruit.py

Folder "SIFT" : includes classical classification algorithms to serve as a baseline / reference

Fruit Classification tutorial:

1.	 Please open "fruit.py" in the folder "fruit" 

2.	 The script is already set up, there are a few parameters that one can finetune, line 26-29
	(Dimensions above 500 crashed our system, please use lower dimensions)

3.	 There are 2 models (DNN, CNN) included, DNN starts from line 107 ends on line 134, and initially commented out,
	if You are interested in how it performs, You just need to uncomment it.
	CNN starts from line 153 ends on line 201

4.	 After running the script, soon one image should pop up, after closing it the same image
	should appear, but its color is normalized, this is just for demonstration purposes.

5. 	 After both pictures are closed the model will be built and evaluated, a confusion matrix
	will pop up with the accuracy results.